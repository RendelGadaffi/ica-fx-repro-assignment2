"""Regression stage: the five regressors used by Kwak et al. (2008), Sec. 4.

- MLP: one hidden layer, hidden nodes in {2,4,6,8,10,15,20}, tanh hidden
  activation (paper uses tansig), linear output (purelin).  The paper trains
  with Levenberg-Marquardt (trainlm); we use sklearn's lbfgs, the closest
  quasi-Newton analogue, over 3 inits and keep the best test RMSE per hidden
  size (the paper selects the best hidden size on the test data).
- SVM: SVR, polynomial kernels of degree 1, 2, 3, all other parameters at
  library defaults (C=1, epsilon=0.1, coef0=0, gamma='scale'), best on test.
- k-NN: weighted regression with kernel phi(z, z_i) = 1/(1 + ||z - z_i||),
  k in {1, 3, 5} (Eq. 16 of the paper).

All inputs/targets are standardized with training statistics; RMSE is reported
on the ORIGINAL target scale.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

HIDDEN = [2, 4, 6, 8, 10, 15, 20]
DEGREES = [1, 2, 3]
KNN_KS = [1, 3, 5]
MLP_SEEDS = [0]


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def knn_weighted(Xtr, ytr, Xte, k):
    """Weighted k-NN regression, Eq. (16): phi(z, z_i) = 1 / (1 + ||z - z_i||)."""
    d = np.linalg.norm(Xte[:, None, :] - Xtr[None, :, :], axis=2)
    idx = np.argsort(d, axis=1)[:, :k]
    w = 1.0 / (1.0 + np.take_along_axis(d, idx, axis=1))
    return (w * ytr[idx]).sum(axis=1) / w.sum(axis=1)


def run_regressors(Xtr, ytr, Xte, yte):
    """Return dict with per-method RMSE on the raw target scale.

    Keys: 'MLP', 'SVM', '1-NN', '3-NN', '5-NN' (standardized features in,
    raw-scale RMSE out).  ytr/yte are raw-scale targets; internal scaling is
    handled here.
    """
    # standardize features
    mu_x, sd_x = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd_x = np.where(sd_x == 0, 1.0, sd_x)
    Xtr_s, Xte_s = (Xtr - mu_x) / sd_x, (Xte - mu_x) / sd_x
    # standardize target
    mu_y, sd_y = ytr.mean(), (ytr.std() if ytr.std() > 0 else 1.0)
    ytr_s, yte_s = (ytr - mu_y) / sd_y, (yte - mu_y) / sd_y

    res = {}
    # MLP: best hidden size on test (paper protocol), best init per size
    best = np.inf
    for h in HIDDEN:
        for seed in MLP_SEEDS:
            m = MLPRegressor(hidden_layer_sizes=(h,), activation="tanh",
                             solver="lbfgs", alpha=1e-4, max_iter=200,
                             tol=1e-5, random_state=seed)
            m.fit(Xtr_s, ytr_s)
            r = _rmse(yte_s, m.predict(Xte_s))
            best = min(best, r)
    res["MLP"] = best * sd_y
    # SVM: best polynomial degree on test
    best = np.inf
    for d in DEGREES:
        m = SVR(kernel="poly", degree=d)
        m.fit(Xtr_s, ytr_s)
        best = min(best, _rmse(yte_s, m.predict(Xte_s)))
    res["SVM"] = best * sd_y
    # k-NN
    for k in KNN_KS:
        res[f"{k}-NN"] = _rmse(yte_s, knn_weighted(Xtr_s, ytr_s, Xte_s, k)) * sd_y
    return res


METHODS5 = ["MLP", "SVM", "1-NN", "3-NN", "5-NN"]


def summarize(res_list):
    """Average/best over the five regression methods, plus std across folds.

    paper_avg  : mean over folds of (mean over the 5 methods)
    paper_std  : mean over the 5 methods of the std across folds
                 (for single-split data sets: std across ICA-FX inits, when present)
    paper_best : mean over folds of (min over the 5 methods)  (table row 2)
    """
    M = np.array([[r[m] for m in METHODS5] for r in res_list])  # (folds, 5)
    avg_per_fold = M.mean(axis=1)
    init_std = [r["_init_std"] for r in res_list if "_init_std" in r]
    if init_std:
        std = float(np.mean(init_std))
    else:
        std = float(M.std(axis=0).mean()) if len(avg_per_fold) > 1 else float("nan")
    return {
        "avg": float(avg_per_fold.mean()),
        "std": std,
        "best": float(M.min(axis=1).mean()),
        "avg_std_per_fold": float(avg_per_fold.std(ddof=1)) if len(avg_per_fold) > 1 else float("nan"),
    }
