"""Run the main experiments of Kwak et al. (2008) and store raw results.

Datasets
  linear    t = 2*x1 + 3*x3 + z, z ~ N(0, 0.1^2), 1000 samples, 10-fold CV
  nonlinear t = sin(x2 + 2*x4) + z, same protocol
  boston    UCI Housing (506 x 13), 10 random 90/10 splits
  oj        Orange Juice NIR (150 train / 68 test, 700 wavelengths),
            from the Wayback Machine; 10 random initializations for ICA-FX

Outputs results/results.json (raw per-cell numbers).

Usage:  python run_all.py [--datasets linear nonlinear boston oj]
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from extract import pca, mlr, sir, phd, icafx, icafx_features, sphering, standardizer, apply_std  # noqa: E402
from regress import run_regressors, summarize, METHODS5  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SEED = 42
ICA_FX_PARAMS = dict(mu1=0.02, mu2=0.02, max_iter=500, tol=1e-6)
EXTRACTORS = {"PCA": pca, "MLR": mlr, "SIR": sir, "PHD": phd}


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------

def load_oj():
    def ld(f):
        return np.loadtxt(os.path.join(DATA, f))
    return ld("OJ_x_learning.txt"), ld("OJ_y_learning.txt"), ld("OJ_x_test.txt"), ld("OJ_y_test.txt")


def load_boston():
    path = os.path.join(DATA, "housing.data")
    if not os.path.exists(path):
        import urllib.request
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
        urllib.request.urlretrieve(url, path)
    raw = np.loadtxt(path)
    return raw[:, :13], raw[:, 13]


def make_artificial(kind, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    z = rng.normal(0, 0.1, n)
    if kind == "linear":
        t = 2 * X[:, 0] + 3 * X[:, 2] + z
    elif kind == "nonlinear":
        t = np.sin(X[:, 1] + 2 * X[:, 3]) + z
    return X, t


def get_dataset(name):
    """Return dict: splits (list of index pairs), X, t, feat_counts, sphering flag, n_inits."""
    if name in ("linear", "nonlinear"):
        X, t = make_artificial(name)
        splits = list(KFold(10, shuffle=True, random_state=SEED).split(X))
        fc = {m: [1, 2, 3, 4, 5] for m in ["PCA", "SIR", "PHD", "ICA-FX"]}
        fc["MLR"] = [1]
        fc["Original"] = [5]
        return dict(X=X, t=t, splits=splits, fc=fc, sphere=False, n_inits=1)
    if name == "boston":
        X, t = load_boston()
        splits = [(np.array(a), np.array(b))
                  for a, b in train_test_split(np.arange(len(X)), test_size=0.1,
                                               train_size=0.9, random_state=SEED,
                                               shuffle=True)]
        for s in range(1, 10):
            a, b = train_test_split(np.arange(len(X)), test_size=0.1, train_size=0.9,
                                    random_state=SEED + s, shuffle=True)
            splits.append((a, b))
        fc = {m: [1, 3, 5, 7, 9, 11, 13] for m in ["PCA", "SIR", "PHD", "ICA-FX"]}
        fc["MLR"] = [1]
        fc["Original"] = [13]
        return dict(X=X, t=t, splits=splits, fc=fc, sphere=False, n_inits=1)
    if name == "oj":
        Xtr, ytr, Xte, yte = load_oj()
        splits = [(np.arange(len(Xtr)), np.arange(len(Xte)))]
        # keep train/test order: store as index pairs into concatenated arrays
        X = np.vstack([Xtr, Xte]); t = np.concatenate([ytr, yte])
        fc = {m: [1, 5, 9, 13] for m in ["PCA", "SIR", "PHD", "ICA-FX"]}
        fc["MLR"] = [1]
        fc["Original"] = [700]
        return dict(X=X, t=t, splits=splits, fc=fc, sphere=True, n_inits=10)
    raise ValueError(name)


# --------------------------------------------------------------------------
# single cell: method x M over all splits
# --------------------------------------------------------------------------

def run_cell(ds_name, method, M, ds, verbose=False):
    X, t = ds["X"], ds["t"]
    n_inits = ds["n_inits"] if method == "ICA-FX" else 1
    res_list = []
    for split_idx, (tr, te) in enumerate(ds["splits"]):
        Xtr, Xte, ytr, yte = X[tr], X[te], t[tr], t[te]
        mu, sd = standardizer(Xtr)
        Xtr_s, Xte_s = apply_std(Xtr, mu, sd), apply_std(Xte, mu, sd)
        tmu, tsd = ytr.mean(), (ytr.std() if ytr.std() > 0 else 1.0)
        ttr_s = (ytr - tmu) / tsd

        if method == "Original":
            Ftr, Fte = Xtr_s, Xte_s
        else:
            if ds["sphere"]:
                S = sphering(Xtr_s, reg=1e-9, n_keep=len(Xtr) - 1)
                Xtr_s, Xte_s = Xtr_s @ S, Xte_s @ S
            if method == "ICA-FX":
                Ftr_l, Fte_l = [], []
                for init in range(n_inits):
                    rng = np.random.default_rng(1000 * (split_idx + 1) + init)
                    f_tr, model = icafx(Xtr_s, ttr_s, M, rng=rng, return_model=True,
                                        **ICA_FX_PARAMS)
                    Ftr_l.append((f_tr, icafx_features(model, Xte_s)))
                # average the 5-method RMSE over inits (std across inits kept in JSON)
                per_init = []
                for f_tr, f_te in Ftr_l:
                    per_init.append(run_regressors(f_tr, ytr, f_te, yte))
                res_list.append(_blend_inits(per_init))
                continue
            Ftr = EXTRACTORS[method](Xtr_s, ttr_s, M)
            Fte = _linear_apply(method, Xtr_s, ttr_s, Xte_s, M)
        res_list.append(run_regressors(Ftr, ytr, Fte, yte))

    s = summarize(res_list)
    return s


def _blend_inits(per_init):
    """Average 5-method RMSEs over ICA-FX initializations (std across inits)."""
    n = len(per_init)
    avg_per_init = np.array([[r[m] for m in METHODS5] for r in per_init])  # (inits, 5)
    mean_r = avg_per_init.mean(axis=0)
    out = {m: float(mean_r[i]) for i, m in enumerate(METHODS5)}
    out["_init_std"] = float(avg_per_init.mean(axis=1).std(ddof=1))
    return out


def _linear_apply(method, Xtr_s, ttr_s, Xte_s, M):
    """Apply the fitted linear extraction map to test data.

    PCA: V from training SVD.  MLR: OLS weights.  SIR/PHD: direction matrices.
    Re-fits the (deterministic) map on training data and reuses its directions.
    """
    if method == "PCA":
        Xc = Xtr_s - Xtr_s.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Xte_s @ Vt[:M].T
    if method == "MLR":
        w, *_ = np.linalg.lstsq(Xtr_s, ttr_s, rcond=None)
        return (Xte_s @ w).reshape(-1, 1)
    if method == "SIR":
        # replicate sir() to obtain direction matrix V
        from scipy import linalg
        Xc = Xtr_s - Xtr_s.mean(axis=0)
        n = len(ttr_s)
        order = np.argsort(ttr_s)
        slices = [s for s in np.array_split(order, 10) if len(s) >= 2]
        mean_all = Xc.mean(axis=0)
        B = np.zeros((Xc.shape[1],) * 2)
        for s in slices:
            f = Xc[s].mean(axis=0)
            B += (len(s) / n) * np.outer(f - mean_all, f - mean_all)
        Sx = np.cov(Xc, rowvar=False) + 1e-10 * np.eye(Xc.shape[1])
        vals, vecs = linalg.eigh(B, Sx)
        V = vecs[:, np.argsort(vals)[::-1][:M]]
        for j in range(V.shape[1]):
            sN = V[:, j] @ Sx @ V[:, j]
            if sN > 0:
                V[:, j] /= np.sqrt(sN)
        return Xte_s @ V
    if method == "PHD":
        from scipy import linalg
        Xc = Xtr_s - Xtr_s.mean(axis=0)
        n, d = Xc.shape
        tc = ttr_s - ttr_s.mean()
        Mm = np.einsum("lj,lk,l->jk", Xc, Xc, tc) / n
        Sx = np.cov(Xc, rowvar=False) + 1e-10 * np.eye(d)
        evals, evecs = np.linalg.eigh(Sx)
        Sinv_half = evecs @ np.diag(1 / np.sqrt(np.maximum(evals, 1e-10))) @ evecs.T
        U, sv, Vt = np.linalg.svd(Sinv_half @ Mm @ Sinv_half, full_matrices=False)
        V = Sinv_half @ U[:, :M]
        for j in range(V.shape[1]):
            sN = V[:, j] @ Sx @ V[:, j]
            if sN > 0:
                V[:, j] = V[:, j] / np.sqrt(sN)
        return Xte_s @ V
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["linear", "nonlinear", "boston", "oj"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "results.json"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_results = {}
    for ds_name in args.datasets:
        ds = get_dataset(ds_name)
        jobs = []
        for method, Ms in ds["fc"].items():
            for M in Ms:
                jobs.append((method, M))
        print(f"[{ds_name}] {len(jobs)} cells", flush=True)
        out = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(run_cell)(ds_name, m, M, ds) for m, M in jobs)
        all_results[ds_name] = {f"{m}|{M}": r for (m, M), r in zip(jobs, out)}
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
