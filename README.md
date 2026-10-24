# ICA-FX for Regression — Paper Reproduction (ICA-FX)

Reproduction of the main experiments of:

> N. Kwak, C. Kim, H. Kim, *Dimensionality reduction based on ICA for regression problems*,
> Neurocomputing 71 (2008) 2596–2603.

Course assignment (data preparation, Oct 2026). This repository contains the code, the
original paper, the data sets, and the raw results of our runs.

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Run

```bash
./venv/bin/python run_all.py --datasets linear nonlinear boston oj --jobs 8
./venv/bin/python src/compare.py     # builds results/comparison.{json,md}
```

`run_all.py` writes `results/results.json` (raw RMSE numbers for every
method × number-of-features cell of the paper's Tables 1–4).
The Orange Juice data files (`data/OJ_*.txt`) were retrieved from the Wayback Machine
from the authors' cited UCL source (snapshot 2004-12-28,
`http://www.dice.ucl.ac.be/mlg/DataBases/ORANGE_JUICE/`); Boston Housing is downloaded
automatically from the UCI repository on first use.

## What is implemented

- Feature extraction: **PCA**, **MLR**, **SIR** (10 slices), **PHD**, and **ICA-FX**
  (Eq. 7–15 of the paper: online natural-gradient updates, µ1 = µ2 = 0.02, 500 iterations,
  log-likelihood stop at 1e-6).
- Regressors (paper Sec. 4): **MLP** (one hidden layer, nodes {2,4,6,8,10,15,20}, tanh /
  linear, best on test), **SVM** (poly kernels 1, 2, 3, default C=1, ε=0.1), **weighted
  k-NN** (Eq. 16, k ∈ {1,3,5}).
- Protocols: 10-fold CV (artificial), 10 random 90/10 splits (Boston), fixed 150/68 split
  with 10 random ICA-FX initializations (Orange Juice). RMSE is reported on the original
  target scale; each table cell is the mean of the five regressors (second row: best).

## Faithfulness notes (where the paper is under-specified)

1. **Score function.** Eq. (6) defines the score as p′/p of an unspecified source density.
   The extended-Infomax form (1+k)·tanh(u) − u with k = sign(E[u·tanh u]) has no stable
   fixed point for the **Gaussian** sources the paper uses in its artificial experiments:
   the target weight v_a is driven to zero (the information channel from t to the features
   closes) and w locks onto the amplified initialization noise. We verified convergence to a
   symmetric attractor w ∝ (1,…,1) from random, identity and zero initializations, and for
   sign- and continuous-k variants and for large v_a. Our implementation therefore uses the
   canonical tanh Infomax score, which keeps the target channel open. The paper's published
   MATLAB code was never released, so the exact score/initialization used by the authors
   cannot be recovered; this is the main source of the ICA-FX deviations reported in the
   comparison tables (see `results/comparison.md`, cells marked ✗ = |ours − paper| > 3·SD).
2. **No whitening of the augmented input.** Whitening [x, t] makes the standardized target
   orthogonal to the inputs (for the linear toy problem t is 99.9% predictable from x),
   which destroys the signal channel; the paper specifies standardization ("normalized
   input vector") and applies sphering only to the small-sample Orange Juice data (we
   follow that, with a small ridge regularization λ = 1e-9 on the covariance).
3. **Numerical stability.** Row normalization of W during the online updates (the scale of
   u is arbitrary in ICA); the 500-epoch, µ = 0.02 schedule is kept exactly as published.
4. **Solvers.** The paper's MLP uses Levenberg–Marquardt (MATLAB); we use sklearn's L-BFGS
   quasi-Newton with three inits per hidden size, best-on-test selection as in the paper.
   SVM defaults are sklearn's (libsvm-compatible C/ε/coef0).
5. **Not specified in the paper:** number of SIR slices (we use the standard 10), the k-NN
   kernel exponent κ, the sphering regularization, the MLP weight init, random seeds
   (ours: 42; all runs reproducible via `SEED` in `run_all.py`).

## Results

See `results/` — `results.json` (raw numbers), `comparison.md` / `comparison.json`
(ours vs. paper, with 3σ flags). The report (PDF/DOCX) summarizes the findings:
the four baseline methods reproduce the published values within ≈1 SD, confirming the
shared pipeline; ICA-FX reproduces qualitatively (features partially aligned with the
informative direction) with the deviations explained above.
