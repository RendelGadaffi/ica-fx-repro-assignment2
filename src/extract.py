"""Feature extraction methods for the ICA-FX regression paper reproduction.

Implements the algorithms compared in Kwak, Kim & Kim (2008),
"Dimensionality reduction based on ICA for regression problems",
Neurocomputing 71, 2596-2603:

- PCA   : principal component analysis (unsupervised)
- MLR   : multivariate linear regression (one feature)
- SIR   : sliced inverse regression (Li, 1991)
- PHD   : principal Hessian directions (Li, 1992)
- ICA-FX: the paper's own method (Eq. (9)-(15), extended Infomax)

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
# ICA-FX for regression (Kwak et al. 2008, Section 3)
# ----------------------------------------------------------------------------

def _extended_score(U, k):
    """Extended-Infomax score function J(u) = (1+k) * tanh(u) - u, k = sign(E[u tanh u]).

    k is estimated per component (Lee, Girolami & Sejnowski 1999).
    """
    return (1.0 + k) * np.tanh(U) - U


def _loglik(W, Phi, U, k):
    """Extended-Infomax log-likelihood of Eq. (12) up to a constant.

    log p(u) ~ (1+k) * log cosh(u) - u^2 / 2
    """
    n = U.shape[0]
    sign, logdet = np.linalg.slogdet(W)
    if sign <= 0:
        return -np.inf
    lp = np.sum((1.0 + k) * np.log(np.cosh(np.clip(U, -50, 50))) - 0.5 * U ** 2)
    return n * logdet + lp


def icafx(X, t, M, mu1=0.02, mu2=0.02, max_iter=500, tol=1e-6,
          rng=None, return_model=False):
    """ICA-FX for regression (Eq. 7-15 of the paper).

    Structure (Eq. 7): u = W x + V t with
      - rows 1..M   of W: feature rows, V_a = V[1..M] unconstrained
      - rows M+1..N : W rows with zero target columns (features independent of t)
      - rows N+1..N+1: identity rows carrying the (standardized) target
    Online (per-sample) natural-gradient updates (Eq. 14-15), 500 epochs:
      W <- W + mu1 (W - phi(u) (u^T W)) ;  V_a <- V_a + mu2 phi_a(u) t^T
    Score phi(u) = tanh(u): Eq. (6) defines the score as p_i'/p_i of the
    source density; the paper's artificial sources are Gaussian and the
    extended-Infomax k-adaptation (phi = (1+k)tanh(u) - u) drives the target
    weight v_a to zero for such symmetric sources (see README "Faithfulness
    notes"), so the canonical tanh Infomax score is used.  Termination:
    max_iter (500) epochs or |change in log-likelihood (Eq. 12)| < 1e-6
    (checked per epoch).  Row normalization of W is a numerical-stability
    device (the scale of u is arbitrary in ICA).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n, N = X.shape
    Nt = 1
    # center the augmented input [x, t] (inputs arrive standardized; NO whitening:
    # whitening the augmented matrix removes the linear x-t correlation that
    # carries the target information, see README "Faithfulness notes")
    Z = np.hstack([X, t.reshape(-1, 1)])          # augmented input [x, t]
    Zw = Z - Z.mean(axis=0)

    W = np.zeros((N + Nt, N + Nt))
    # Signal-matched initialization: the linearized gradient of the log
    # likelihood (Eq. 13) at small weights is proportional to
    # w + v * cov(t, x); the steepest-ascent direction for a feature row is
    # therefore (w, v) ∝ (cov(t, x), 1).  Rows M+1..N (the discarded features
    # f_b) start at zero: an identity initialization pulls all rows onto a
    # shared (1,...,1) attractor through the W^T W coupling of the natural
    # gradient and destroys the signal alignment (see README "Faithfulness
    # notes"; the paper specifies random initialization but gives no details).
    cov_tx = (Zw[:, N] @ Zw[:, :N]) / n            # cov(t, x) per input dim
    Va = np.zeros((M, Nt))
    for m in range(M):
        s = 1.0 if rng.standard_normal() > 0 else -1.0
        Va[m] = 1.5 * s
        W[m, :N] = 1.5 * s * cov_tx + 0.02 * rng.standard_normal(N)
    W[:M, N:] = Va
    W[N:, :] = np.eye(N + Nt)[N:]
    tc = t - t.mean()

    L_prev = -np.inf
    for _ in range(max_iter):
        order = rng.permutation(n)
        for l in order:
            u = W @ Zw[l]                                   # (N+Nt,)
            phi = np.tanh(u)                                # score, Eq. (6)
            g = W.T @ u                                     # u^T W (natural gradient)
            # Eq. 15: online natural-gradient update, structure-aware
            W[:M] += mu1 * (W[:M] - np.outer(phi[:M], g)) + mu2 * np.outer(phi[:M], tc[l])
            W[M:N] += mu1 * (W[M:N] - np.outer(phi[M:N], g))
            # numerical stability: keep the scale of each u_i fixed (the scale
            # is arbitrary in ICA; pinning it keeps the score function in its
            # informative regime and prevents super-Gaussian blow-up)
            nrm = np.linalg.norm(W[:N], axis=1, keepdims=True)
            W[:N] /= np.maximum(nrm, 1e-12)
        # re-impose the structure of Eq. (7)
        Va = W[:M, N:].copy()
        W[M:N, N:] = 0.0
        W[N:, :] = np.eye(N + Nt)[N:]

        U = Zw @ W.T
        Phi = np.tanh(U)
        L = _loglik(W, Phi, U, 0.0)
        if np.isfinite(L) and np.isfinite(L_prev) and abs(L - L_prev) < tol:
            break
        L_prev = L

    # deployment features f_a = W_a x (Eq. 10 with the target term removed:
    # the target is not available at test time, so it must not enter the
    # features used to train the regressors either)
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
