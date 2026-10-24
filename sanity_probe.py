"""Sanity checks for icafx with the random init.

Linear toy M=1: cos with the informative direction [2,0,3,0,0] ~ 0.999.
Boston M=1: finite, cos with OLS direction ~ 1.
OJ M=13: finite.
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
warnings.simplefilter("ignore")
import numpy as np
from extract import icafx, standardizer, apply_std, sphering
from run_all import make_artificial, load_boston, load_oj

def cos(a, b):
    return abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- linear toy ---
X, t = make_artificial("linear")
Xs = (X - X.mean(0)) / X.std(0)
ts = (t - t.mean()) / t.std()
w = icafx(Xs, ts, 1, return_model=True)[1]["W"][0, :5]
print("linear  M=1 |w|=%.3f  cos=%.4f" % (np.linalg.norm(w), cos(w, [2, 0, 3, 0, 0])))

# --- Boston ---
Xb, tb = load_boston()
mu, sd = standardizer(Xb)
Xb_s = apply_std(Xb, mu, sd)
tb_s = (tb - tb.mean()) / tb.std()
wb = icafx(Xb_s, tb_s, 1, return_model=True)[1]["W"][0, :13]
ols = np.linalg.lstsq(Xb_s, tb_s, rcond=None)[0]
print("boston  M=1 |w|=%.3f  finite=%s  cos(OLS)=%.4f"
      % (np.linalg.norm(wb), np.all(np.isfinite(wb)), cos(wb, ols)))

# --- OJ ---
Xtr, ytr, Xte, yte = load_oj()
mu, sd = standardizer(Xtr)
Xtr_s = apply_std(Xtr, mu, sd)
S = sphering(Xtr_s, reg=1e-9, n_keep=len(Xtr) - 1)
Xtr_q = Xtr_s @ S
ytr_s = (ytr - ytr.mean()) / ytr.std()
W = icafx(Xtr_q, ytr_s, 13, rng=np.random.default_rng(7), return_model=True)[1]["W"]
print("oj      M=13 |W|=%.3f  finite=%s" % (np.linalg.norm(W), np.all(np.isfinite(W))))
