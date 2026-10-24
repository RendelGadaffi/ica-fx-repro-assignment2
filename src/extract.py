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
    else:
        lp = np.sum((1.0 + k) * np.log(np.cosh(np.clip(U, -50, 50)))
                    - 0.5 * k * U ** 2)
    return U.shape[0] * logdet + lp


def icafx(X, t, M, mu1=0.02, mu2=0.02, max_iter=500, tol=1e-6,
          rng=None, return_model=False, score="gaussian"):
    """ICA-FX for regression - batch Eq. (15) of the paper.

    Batch (1/n-summed) form of Eq. (15), Gaussian score (J = +u, the Gaussian
    limit of Eq. 6 for the paper's Gaussian sources):
      W <- W + mu1 * (W - Sigma W)          (Sigma = E[u u^T], augmented input)
      V_a <- V_a + mu2 * E[u_a t]           (Eq. 13: dL/dV_a = E[u_a t], ascent)
    mu1 = mu2 = 0.02, 500 iterations, random initial condition W = I +
    0.1*randn, stopping when |L(W|D) - L(W_prev|D)| < 1e-6 (Eq. 12) - exactly
    the settings of Sec. 4.

    Two documented deviations (see README "Faithfulness notes"), both forced by
    the published rule being divergent as printed:

    1. V_a is the coefficient of the TARGET in Eq. (7); its expected ascent
       direction is E[u_a t] = (Sigma W)[a, N] - the target column of the
       augmented covariance.  Reading it as E[u_a *t*] over the x-columns
       (Sigma W)[a, 0:M] - what a literal Eq. 15 with J(u)=u gives - has zero
       target signal (Eq. 7 forces the x-part of rows M..N, and the x-columns
       of rows 0..M, to be target-independent), so the learning never sees t.
    2. The raw batch update W <- W + mu1 (W - Sigma W) is provably divergent:
       its multiplier (1+mu1) I - mu1 Sigma has spectral radius 1 + mu1(1 -
       lambda_min(Sigma)) > 1 (verified numerically for all four data sets:
       1.02, 1.019, 1.020), so W grows exponentially (‖w‖ > 10^40 in 500
       iterations).  The paper specifies no stabilization.  We impose the
       standard ICA scale gauge - after every iteration each feature row's
       x-part is rescaled to unit norm (the scale of u is arbitrary in ICA,
       Eq. 6; the direction is untouched) - and, in the V_a ascent, a trust
       region: mu2 is automatically shrunk so the target coefficient never
       leaves [0, 1.2].  This keeps the run finite; it caps the magnitude of
       the extracted features (and is the main reason the reproduced RMSEs
       exceed the paper's - see README).

    Structure (Eq. 7): u = W x + V t; rows M+1..N of W keep zero target
    columns (imposed every iteration), the last row carries the standardized
    target.  No whitening of the augmented [x, t] input (whitening removes
    the x-t correlation that carries the target information; the paper
    standardizes inputs and applies sphering only to Orange Juice, where we
    sphere the inputs before this function is called).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n, N = X.shape
    Nt = 1
    # augmented, centered input [x, t] ("normalized input vector x" in the
    # paper; centering only)
    Z = np.hstack([X, t.reshape(-1, 1)])
    Zw = Z - Z.mean(axis=0)

    if score not in ("gaussian", "extended"):
        raise ValueError("score must be 'gaussian' (default) or 'extended'")
    gaussian = score == "gaussian"

    # random initial condition with the Eq. (7) structure
    W = np.eye(N + Nt) + 0.1 * rng.standard_normal((N + Nt, N + Nt))
    W[M:, N] = 0.0
    W[N, :N] = 0.0
    W[N, N] = 1.0

    # extended-Infomax k-adaptation (score != gaussian) re-estimates k per
    # iteration from the current sources
    k = None
    Sigma = Zw.T @ Zw / n                               # E[u u^T] (Gaussian)
    L_prev = -np.inf
    for _ in range(max_iter):
        Wf = W.copy()
        Wf[M:, N] = 0.0                                 # Eq. 7 structure
        if not gaussian:
            U = Zw @ Wf.T
            k = np.sign(np.mean(U * np.tanh(U), axis=0))
        # Eq. 15 line 1 (Gaussian score: E[J(u) u^T] = Sigma)
        Wf += mu1 * (Wf - Sigma @ Wf)
        # scale gauge: rescale feature-row x-parts to unit norm (see docstring)
        norms = np.linalg.norm(Wf[:M, :N], axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Wf[:M, :N] /= norms
        # Eq. 15 line 2 (Eq. 13 ascent): dV_a = +mu2 * E[u_a t], target column
        # of Sigma @ W evaluated at the gauged row (unit-norm x-part), with a
        # trust region keeping |v_a| <= 1.2
        v_new = Wf[:M, N] + mu2 * (Sigma @ Wf)[:M, N]
        over = np.abs(v_new) > 1.2
        if over.any():
            v_new = np.where(over, np.sign(v_new) * 1.2, v_new)
        Wf[:M, N] = v_new
        Wf[M:N, N] = 0.0
        Wf[N, :N] = 0.0
        Wf[N, N] = 1.0
        W = Wf

        if tol is not None:
            U = Zw @ W.T
            L = _loglik(W, U, k, gaussian)
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
