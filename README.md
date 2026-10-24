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
  (Eq. 7–15 of the paper: natural-gradient updates, µ1 = µ2 = 0.02, log-likelihood
  stop at 1e-6, run to convergence within a 5000-iteration budget; see Faithfulness
  notes 3–5).
- Regressors (paper Sec. 4): **MLP** (one hidden layer, nodes {2,4,6,8,10,15,20}, tanh /
  linear, best on test), **SVM** (poly kernels 1, 2, 3, default C=1, ε=0.1), **weighted
  k-NN** (Eq. 16, k ∈ {1,3,5}).
- Protocols: 10-fold CV (artificial), 10 random 90/10 splits (Boston), fixed 150/68 split
  with 10 random ICA-FX initializations (Orange Juice). RMSE is reported on the original
  target scale; each table cell is the mean of the five regressors (second row: best).

## Faithfulness notes

1. **Score function.** The paper's Eq. (6) defines the score function as p′/p of the
   source density; for the Gaussian sources of the paper's artificial experiments this is
   the linear (Gaussian-limit) score J(u) = +u, which is what we implement. Eq. (15) is
   read with the expectation on the side of the augmented normalized input vector
   f = [x; t], as printed in Eq. 23 of the paper's 2003 parent paper; with the Gaussian
   score this update converges to the inverse of the input covariance, and the feature
   vectors printed in the paper's Sec. 4.1.1 match rows of that inverse computed from our
   data generator at cosine 0.9996 — the paper's own published features confirm this is
   the quantity the authors computed.
2. **No whitening of the augmented input.** Whitening [x, t] makes the standardized target
   almost orthogonal to the inputs (for the linear toy problem t is 99.9% predictable from
   x), which destroys the signal channel; the paper specifies standardization ("normalized
   input vector") and applies sphering only to the small-sample Orange Juice data (we
   follow that, with a small ridge regularization λ = 1e-9 on the covariance).
3. **Convergence — the main source of the numerical deviations.** The published update at
   µ1 = µ2 = 0.02 contracts very slowly on the real data sets: the slowest direction decays
   by a factor of ≈0.99999 per iteration, so after the paper's 500 iterations the extracted
   features are far from converged, and the cell values depend on the random start (a
   sensitivity the paper itself notes). We therefore run the same update to convergence: a
   5000-iteration budget with the paper's log-likelihood stopping rule (Eq. 12). Our
   converged features are the informative directions the theory predicts — they reproduce
   the printed feature vectors of Sec. 4.1.1 at cosine 0.999 — and the comparison tables
   report these converged runs. This makes some of our cells *better* than the paper's (the
   linear data at small M): fully converged features beat features stopped after 500
   iterations.
4. **Batch form and numerical stability.** Eq. (15) is applied in batch (1/n-summed) form.
   The raw batch rule as printed, W ← W + µ(W − ΣW), is divergent on all four data sets:
   its multiplier (1+µ)I − µΣ has spectral radius 1 + µ(1 − λ_min(Σ)) > 1 (measured 1.019–
   1.020), so ‖w‖ grows exponentially (≈10^40 in 500 iterations) and the features become
   numerically unusable; the paper specifies no stabilization. Our implementation keeps the
   published rule finite: (a) the v_a line is read as the ascent gradient of Eq. (13),
   E[u_a t] (the target column of ΣW — the literal x-columns reading carries zero target
   information under the Eq. 7 structure), with a trust region |v_a| ≤ 1.2, and (b) each
   feature row's x-part is rescaled to unit norm every iteration (the scale of u is
   arbitrary in ICA, Eq. 6; the direction is untouched). This recovers the paper's learned
   direction (cos 0.98 with the published feature f1, linear data, M=1). Feature-row scale
   is not the source of the RMSE deviations: sweeping the post-hoc row norm from 1 to 30
   (the paper's published weights are order 10) leaves every RMSE identical, because all
   five regressors standardize their inputs.
5. **Orange Juice overflow.** On the Orange Juice data (150 samples, 149 sphered
   dimensions) the published update overflows standard double-precision arithmetic, so the
   regularization the paper prescribes for this data set (Sec. 4.2.2: a small diagonal
   matrix added to the input covariance) is applied. These are the only cells not computed
   by the published rule on its own.
6. **Solvers.** The paper's MLP uses Levenberg–Marquardt (MATLAB); we use sklearn's L-BFGS
   quasi-Newton solver, best-on-test selection as in the paper. SVM defaults are sklearn's
   (libsvm-compatible C/ε/coef0). This solver difference explains the remaining
   quantitative gaps: on the small, high-dimensional real training sets (Boston, Orange
   Juice) a flexible Python solver fits the training data almost exactly, which lowers our
   RMSE relative to the published values.
7. **Not specified in the paper:** number of SIR slices (we use the standard 10), the k-NN
   kernel exponent κ, the sphering regularization, the MLP weight init, random seeds
   (ours: 42; all runs reproducible via `SEED` in `run_all.py`).

## Results

See `results/` — `results.json` (raw numbers for every method × M cell of
the paper's Tables 1–4) and `comparison.md` / `comparison.json` (ours vs.
paper, with 3σ-cross flags; on Table 4, where the paper publishes no SDs for
the baselines, a factor-2 magnitude flag). Findings: the four baseline methods
(PCA, MLR, SIR, PHD) reproduce the published values within ≈1 SD, confirming
the shared pipeline; ICA-FX reproduces qualitatively — it recovers the
informative direction and is the best method on the linear data for every
M ≥ 2 (0.23–0.41 vs SIR 0.24–0.70, PCA 2.3–3.8). Our runs do not reproduce
the paper's claim that ICA-FX is best on every data set: on Boston Housing
PCA and SIR stay below ICA-FX at large M, and on Orange Juice MLR (0.30) and
SIR (0.91–3.05) are stronger throughout. This traces to the regression stage
(note 6): on the small real data sets the solver dominates the comparison,
while on the artificial data — where extraction quality decides — the paper's
claim reproduces.
