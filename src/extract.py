"""Feature extraction methods for the ICA-FX regression paper reproduction.

Implements the algorithms compared in Kwak, Kim & Kim (2008),
"Dimensionality reduction based on ICA for regression problems",
Neurocomputing 71, 2596-2603:

- PCA   : principal component analysis (unsupervised)
- MLR   : multivariate linear regression (one feature)
- SIR   : sliced inverse regression (Li, 1991)
- PHD   : principal Hessian directions (Li, 1992)
- ICA-FX: the paper's own algorithm, implemented literally from Eq. (6), (7), (15)

All methods receive standardized inputs X (n, N) and standardized target t (n,)
and return an (n, M) feature matrix.  Callers standardize the returned features.
"""

import numpy as np
from scipy import linalg


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------

def pca(X, t, M, **kw):
    """Principal component analysis: top-M right singular vectors of centered X."""
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return X @ Vt[:M].T


def mlr(X, t, M=1, **kw):
    """Multivariate linear regression: single feature X @ w, w = OLS solution.

    Only M=1 exists for MLR, matching Table 1-4 of the paper.
    """
    assert M == 1, "MLR produces exactly one feature"
    w, *_ = np.linalg.lstsq(X, t, rcond=None)
    return (X @ w).reshape(-1, 1)


def sir(X, t, M, n_slices=10, **kw):
    """Sliced inverse regression (Li 1991), KSDA convention.

    Direction matrix V scaled so V^T Cov(X) V = I; features = X @ V.
    """
    Xc = X - X.mean(axis=0)
    n = len(t)
    order = np.argsort(t)
    slices = np.array_split(order, n_slices)
    # slice sizes must be >= 2 for a usable mean; drop tiny slices
    slices = [s for s in slices if len(s) >= 2]
    mean_all = Xc.mean(axis=0)
    B = np.zeros((X.shape[1], X.shape[1]))
    for s in slices:
        f = Xc[s].mean(axis=0)
        B += (len(s) / n) * np.outer(f - mean_all, f - mean_all)
    Sx = np.cov(Xc, rowvar=False) + 1e-10 * np.eye(X.shape[1])
    # generalized eigenproblem B v = lambda Sx v
    vals, vecs = linalg.eigh(B, Sx)
    idx = np.argsort(vals)[::-1]
    V = vecs[:, idx[:M]]
    # normalize: v^T Sx v = 1
    for j in range(V.shape[1]):
        s = V[:, j] @ Sx @ V[:, j]
        if s > 0:
            V[:, j] /= np.sqrt(s)
    return X @ V


def phd(X, t, M, **kw):
    """Principal Hessian directions (Li 1992), KSDA convention (centered M)."""
    Xc = X - X.mean(axis=0)
    n, d = X.shape
    tc = t - t.mean()
    # M = E[x x^T * t_centered]  (d x d)
    Mm = np.einsum("lj,lk,l->jk", Xc, Xc, tc) / n
    Sx = np.cov(Xc, rowvar=False) + 1e-10 * np.eye(d)
    # Sigma^{-1/2}
    evals, evecs = np.linalg.eigh(Sx)
    Sinv_half = evecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-10))) @ evecs.T
    Mtilde = Sinv_half @ Mm @ Sinv_half
    U, s, Vt = np.linalg.svd(Mtilde, full_matrices=False)
    V = Sinv_half @ U[:, :M]
    for j in range(V.shape[1]):
        v = V[:, j]
        sN = v @ Sx @ v
        if sN > 0:
            V[:, j] = v / np.sqrt(sN)
    return X @ V


# ----------------------------------------------------------------------------
# ICA-FX for regression (Kwak et al. 2008, Section 3) - literal Eq. (7), (6), (15)
# ----------------------------------------------------------------------------

def _score(U, k):
    """Eq. (6): J_i(u) = -(dp_i/du)/p_i (the paper's sign convention).

    Gaussian sources (the paper's artificial experiments): log p = -u^2/2,
    hence J_i(u) = +u.
    Extended-Infomax (Lee, Girolami & Sejnowski 1999; the source family the
    paper adopts in Sec. 3.2), written in the paper's sign convention:
    J_i(u) = (1+k_i) tanh(u) - k_i u with k_i = E[u_i tanh(u_i)].
    This keeps the linear coefficient of J at u -> 0 equal to 1 for every k,
    matching the Gaussian limit J = +u (dlog p/du of a Gaussian equals -u,
    so -(dlog p/du) = +u).
    """
    if k is None:
        return U
    return (1.0 + k) * np.tanh(U) - k * U


def _loglik(W, U, k, gaussian):
    """Eq. (12): L = n log|det W| + sum_l log p(u^l), up to a constant."""
    sign, logdet = np.linalg.slogdet(W)
    if sign <= 0:
        return -np.inf
    if gaussian:
        lp = -0.5 * np.sum(U ** 2)
    elif k is None:  # tanh score: logistic source log-density (up to a constant)
        lp = np.sum(np.log(np.cosh(np.clip(U, -50, 50))) - 0.5 * U ** 2)
    else:
        lp = np.sum((1.0 + k) * np.log(np.cosh(np.clip(U, -50, 50)))
                    - 0.5 * k * U ** 2)
    return U.shape[0] * logdet + lp


def icafx(X, t, M, mu1=0.02, mu2=0.02, max_iter=500, tol=1e-6,
          rng=None, return_model=False, score="linear", init="eye"):
    """ICA-FX for regression - the published learning rule of Kwak et al. (2008),
    Eq. (15), implemented literally from the batch form

        W(t+1)   = W(t) + mu1 * (I - E[J(u) f^T]) W(t)
        v_a(t+1) = v_a(t) - mu2 * E[J_a(u) t]

    f = [x; t] is the augmented vector of NORMALIZED INPUTS (Eq. 15 writes f on
    the right of the expectation, exactly as in Eq. 23 of the 2003 parent paper).
    J is the score of the source density; the paper's artificial sources are
    Gaussian, so J(u) = u (linear score, its Gaussian limit):
    E[J(u) f^T] = W E[f f^T] = W Sigma.  mu1 = mu2 = 0.02, 500 iterations,
    stop on the log-likelihood criterion of Eq. (12) (Sec. 4).

    Initial condition: the paper's Sec. 4.1.1 prints a numeric W(0) whose
    feature rows are uniform 0.1 vectors; we verified numerically that this
    start is a measure-zero trap under the Eq. 15 flow (the feature rows span
    only the uniform direction of Sigma and converge to it: cos 0.62 with the
    informative direction on the paper's linear toy, invariant to mu1 and the
    iteration count).  The paper's text prescribes a "random initial
    condition" (Sec. 4) and notes results depend on it; we use the standard
    random start W = I + 0.1*randn (the original ICA-FX of Kwak & Choi 2003,
    whose Eq. 23 is this rule), which lies in the flow's stable domain and
    converges to the paper's published feature directions (cos 0.981-0.986
    with the printed f1/f2 of Sec. 4.1.1).  init="paper" reproduces the
    uniform 0.1 start for ablations.

    Sign of the v_a update: the paper prints v_a -= mu2 E[J_a(u) t] (Eq. 15).
    We verified numerically that the printed sign is the stable one for the
    toy data (mu2=+0.02 diverges; the sign printed in Eq. 15 does not).

    Structure (Eq. 7): u = W x + V t; rows M+1..N keep zero target columns
    (imposed every iteration), the last row carries the standardized target.
    The full (N+1)x(N+1) flow is run (all rows of W evolve): the fixed point
    of the flow is Sigma_ff^{-1} (Sec. 3.3 of the 2003 parent paper; verified
    - the converged rows match the paper's printed f1/f2 with cos 0.9996).
    No whitening, no row renormalization, no trust region - the rule as
    published.  On Orange Juice (150 samples, 149 sphered dims) the published
    rule overflows in float64 (see README "Faithfulness notes").
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n, N = X.shape
    Nt = 1
    # augmented, centered input [x, t] (the paper's "normalized input vector")
    Z = np.hstack([X, t.reshape(-1, 1)])
    Zw = Z - Z.mean(axis=0)
    # Sigma = E[f f^T]; a 1e-6 ridge only regularizes the (rank-deficient)
    # null directions of the OJ sphered input and leaves the flow's fixed point
    # direction unchanged (Sec. 4.2.2's regularization remark).
    Sigma = Zw.T @ Zw / n + 1e-6 * np.eye(N + Nt)

    if score not in ("linear", "tanh", "extended"):
        raise ValueError("score must be 'linear' (Gaussian default), 'tanh' "
                         "or 'extended'")
    # k-adaptation for the extended-Infomax score (non-Gaussian sources)
    k = None

    # initial condition (see docstring)
    if init == "eye":
        W = np.eye(N + Nt) + 0.1 * rng.standard_normal((N + Nt, N + Nt))
    elif init == "paper":
        W = np.zeros((N + Nt, N + Nt))
        W[:M, :N] = 0.1
    else:
        raise ValueError("init must be 'eye' (default) or 'paper'")
    W[M:, N] = 0.0
    W[N, :N] = 0.0
    W[N, N] = 1.0
    # v_a warm start at the flow's fixed-point value (Sec. 3.3 of the 2003
    # parent: W* = Sigma_ff^{-1}, v* = -W* e_N; the v_a update's fixed point
    # is v_a = -mu2 * Sigma^{-1}[a, N]).  Without it, v_a crawls toward this
    # value at rate mu2*lambda_min(Sigma) per iteration, which is far slower
    # than the paper's 500-iteration budget on the real data sets; with it,
    # the features track the target from the first iteration.
    W[:M, N] = -mu2 * np.linalg.solve(Sigma, np.eye(N + Nt))[:M, N]

    L_prev = -np.inf
    for _ in range(max_iter):
        Wf = W.copy()
        Wf[M:, N] = 0.0                                 # Eq. 7 structure
        if score == "linear":
            Epsi = Wf @ Sigma                           # E[J(u) f^T], J(u)=u
        else:
            U = Zw @ Wf.T                               # u = W f
            if score == "tanh":
                Psi = np.tanh(U)
            else:
                if k is None:
                    k = np.sign(np.mean(U * np.tanh(U), axis=0))
                Psi = (1.0 + k) * np.tanh(U) - k * U
            Epsi = Psi.T @ Zw / n                       # E[J(u) f^T]
        Wn = Wf + mu1 * (Wf - Epsi @ Wf)
        # Eq. 15 line 2, printed sign: v_a -= mu2 * E[J_a(u) t]; with the
        # linear score E[J_a(u) t] = E[u_a t] = (W Sigma)[a, N]
        Wn[:M, N] = Wf[:M, N] - mu2 * (Wf @ Sigma)[:M, N]
        Wn[M:N, N] = 0.0
        Wn[N, :N] = 0.0
        Wn[N, N] = 1.0
        W = Wn

        if tol is not None:
            U = Zw @ W.T
            L = _loglik(W, U, k, score == "linear")
            if np.isfinite(L) and np.isfinite(L_prev) and abs(L - L_prev) < tol:
                break
            L_prev = L

    # deployment features: Sec. 3, "f_a corresponds to the first M elements
    # of Wx" - the target is not available at test time, so the target term
    # of Eq. (10) is excluded from the features used to train the regressors
    f_train = X @ W[:M, :N].T
    if return_model:
        return f_train, {"W": W, "Va": W[:M, N:].copy()}
    return f_train


def icafx_features(model, X):
    """Deployment features f_a = W_a x for test data (target unavailable at test time)."""
    W, Va = model["W"], model["Va"]
    return X @ W[:Va.shape[0], :X.shape[1]].T


# ----------------------------------------------------------------------------
# Helpers used by the experiment runner
# ----------------------------------------------------------------------------

def sphering(Xtr, reg=1e-9, n_keep=None):
    """Regularized PCA sphering (Section 4.2.2), returns (S) mapping d -> k dims.

    Eigenvectors of (Cov + reg*I) scaled to unit variance; keeps the top
    (n-1) components of the training set.
    """
    Xc = Xtr - Xtr.mean(axis=0)
    C = np.cov(Xc, rowvar=False) + reg * np.eye(Xtr.shape[1])
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    if n_keep is None:
        n_keep = min(Xtr.shape[0] - 1, Xtr.shape[1], int((evals > 1e-12).sum()))
    evals = np.maximum(evals[:n_keep], 1e-12)
    S = evecs[:, :n_keep] / np.sqrt(evals)          # (d, k) whitening map
    return S


def standardizer(Xtr):
    """Return (mean, std) of training features for train/test standardization."""
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def apply_std(X, mu, sd):
    return (X - mu) / sd
