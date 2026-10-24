"""Verification of the README claims about the ICA-FX implementation.

Checks (all on the paper's linear toy data, t = 2*x1 + 3*x3 + noise):
  1. The published feature direction f1 ~ 2*x1 + 3*x3 (Sec. 4.1.1) is
     reproduced by the converged batch Eq. (15) update (cos > 0.99).
  2. The converged update W(5000) rows match the inverse of the augmented
     input covariance Sigma (the flow's fixed point, Sec. 3.3 of the 2003
     parent paper): cos > 0.99.
  3. At the published 500-iteration budget the cell is NOT converged:
     the 500-iteration RMSE (avg over the five regressors) is markedly
     worse than the converged (5000-iteration budget) value.

Run:  ./venv/bin/python verify_claims.py
"""
import sys
import os
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
warnings.simplefilter("ignore")

from extract import icafx  # noqa: E402
from run_all import make_artificial, load_oj  # noqa: E402
from regress import run_regressors  # noqa: E402

F1_PAPER = np.array([2.0, 0.0, 3.0, 0.0, 0.0])   # Sec. 4.1.1: f1 = 2x1 + 3x3
METHODS = ("MLP", "SVM", "1-NN", "3-NN", "5-NN")


def cos(a, b):
    return abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    X, t = make_artificial("linear")
    Xs = (X - X.mean(0)) / X.std(0)
    ts = (t - t.mean()) / t.std()

    # 1) converged published-rule update recovers f1's direction
    f, model = icafx(Xs, ts, 1, rng=np.random.default_rng(0),
                     return_model=True, max_iter=5000, tol=1e-6)
    w = model["W"][0, :5]
    c1 = cos(w, F1_PAPER)

    # 2) fixed point of the flow: row of Sigma^{-1} (augmented, centered)
    Z = np.hstack([Xs, ts.reshape(-1, 1)])
    Zw = Z - Z.mean(0)
    Sigma = Zw.T @ Zw / len(Z)
    c2 = cos(np.linalg.inv(Sigma)[0, :5], F1_PAPER)
    c3 = cos(w, np.linalg.inv(Sigma)[0, :5])

    # 3) 500 iterations (paper budget) vs converged: cell quality
    f500 = icafx(Xs, ts, 1, rng=np.random.default_rng(0), max_iter=500)
    # train/test = full data (the toy cell reports in-sample CV RMSE, see run_all)
    r500 = run_regressors(f500, t, f500, t)
    rfull = run_regressors(f, t, f, t)
    avg500 = np.mean([r500[m] for m in METHODS])
    avgfull = np.mean([rfull[m] for m in METHODS])

    print(f"cos(converged w, published f1)          = {c1:.4f}")
    print(f"cos(Sigma^-1 row, published f1)         = {c2:.4f}")
    print(f"cos(converged w, Sigma^-1 row)          = {c3:.4f}")
    print(f"linear M=1 avg RMSE @500 iters          = {avg500:.3f}")
    print(f"linear M=1 avg RMSE converged (5000)    = {avgfull:.3f}")

    ok = c1 > 0.99 and c2 > 0.99 and c3 > 0.99 and avg500 > 2 * avgfull
    print("ALL CLAIMS VERIFIED" if ok else "MISMATCH - claims NOT verified")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
