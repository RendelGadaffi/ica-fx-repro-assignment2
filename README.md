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
`--datasets` selects data sets (any subset of `linear nonlinear boston oj`);
`--jobs` sets the number of parallel workers. Results for the datasets you
run are merged into the existing `results.json`.
The Orange Juice data files (`data/OJ_*.txt`) were retrieved from the Wayback Machine
from the authors' cited UCL source (snapshot 2004-12-28,
`http://www.dice.ucl.ac.be/mlg/DataBases/ORANGE_JUICE/`); Boston Housing is downloaded
automatically from the UCI repository on first use.

### Parameters you can change

All defaults follow the paper (Sec. 4.1.1 / Sec. 4). Every setting lives in the code:

| Parameter | Default (paper setting) | Where |
|---|---|---|
| `mu1`, `mu2` (learning rates) | 0.02, 0.02 | `run_all.py`: `ICA_FX_PARAMS`, `ICA_FX_PARAMS_OJ` |
| `max_iter`, `tol` (convergence) | 5000 / 1e-6 (Tables 1–3); 500 / 1e-6 (OJ, Sec. 4.2.2 variant) | `run_all.py`: `ICA_FX_PARAMS`, `ICA_FX_PARAMS_OJ` |
| `M` (features extracted) | linear/nonlinear 1–5; Boston 1–13; OJ 1,5,9,13 | `run_all.py`: `get_dataset` (`fc` dicts) |
| Random seed | 42 (all splits and inits; ICA-FX inits derive as `1000·split + init`) | `run_all.py`: `SEED` |
| Train/test protocol | 10-fold CV; ten 90/10 splits; fixed 150/68 (OJ) | `run_all.py`: `get_dataset` (`splits`) |
| SIR slices | 10 (unspecified in paper) | `src/extract.py`: `sir(n_slices=...)` |
| OJ sphering ridge | 1e-9 | `run_all.py`: `run_cell` (`sphering(reg=...)`) |
| MLP hidden nodes / inits | {2,4,6,8,10,15,20} / 5 seeds, best on test | `src/regress.py`: `HIDDEN`, `MLP_SEEDS` |
| SVM kernel degrees / defaults | poly degree 1/2/3 (`DEGREES`); C=1, ε=0.1, coef0=0, gamma='scale' | `src/regress.py`: `DEGREES`; sklearn defaults |
| k-NN k values (Eq. 16) | {1,3,5} | `src/regress.py`: `KNN_KS` |

`verify_claims.py` checks the quantitative claims in the notes below.

## What is implemented

- Feature extraction: **PCA**, **MLR**, **SIR** (10 slices), **PHD**, and **ICA-FX**
  (Eq. 7–15 of the paper: natural-gradient updates, µ1 = µ2 = 0.02, log-likelihood
  stop at 1e-6, run to convergence within a 5000-iteration budget; Orange Juice uses
  the Sec. 4.2.2 regularized variant, see notes).
- Regressors (paper Sec. 4): **MLP** (one hidden layer, nodes {2,4,6,8,10,15,20}, tanh /
  linear, best on test), **SVM** (poly kernels 1, 2, 3, default C=1, ε = 0.1), **weighted
  k-NN** (Eq. 16, k ∈ {1,3,5}).
- Protocols: 10-fold CV (artificial), 10 random 90/10 splits (Boston), fixed 150/68 split
  with 10 random ICA-FX initializations (Orange Juice). RMSE is reported on the original
  target scale; each table cell is the mean of the five regressors (second row: best).
- `verify_claims.py` checks the quantitative claims in the notes below.

## Results

See `results/` — `results.json` (raw numbers for every method × M cell of
the paper's Tables 1–4) and `comparison.md` / `comparison.json` (ours vs.
paper, with 3σ-cross flags; on Table 4, where the paper publishes no SDs for
the baselines, a factor-2 magnitude flag).

Findings: the four baseline methods (PCA, MLR, SIR, PHD) reproduce the
published values within ≈1 SD, confirming the shared pipeline. ICA-FX
reproduces qualitatively — it recovers the informative direction and is the
best method on the linear data for every M ≥ 2 (0.23–0.41 vs SIR 0.24–0.70,
PCA 2.3–3.8). Our runs do not reproduce the paper's claim that ICA-FX is best
on every data set: on Boston Housing, PCA and SIR stay below ICA-FX at large
M, and on Orange Juice, MLR (0.30) and SIR (0.91–3.05) are stronger
throughout. This traces to the regression stage (note 4): on the small real
data sets the solver dominates the comparison, while on the artificial data
— where extraction quality decides — the paper's claim reproduces.

## Notes on faithfulness

1. **Score function.** Gaussian sources ⇒ the linear score J(u) = +u of Eq. (6),
   implemented with Eq. (15) on the input side (as printed in Eq. 23 of the paper's
   2003 parent paper). The converged features reproduce the paper's printed f1 at
   cosine 0.9995 — verified in `verify_claims.py`.
2. **No whitening of the augmented input** (it would make the target nearly orthogonal
   to the inputs and close the signal channel); the paper specifies standardization and
   sphering for Orange Juice only, which we follow (ridge λ = 1e-9).
3. **Convergence — main source of the numerical deviations.** At µ1 = µ2 = 0.02 the
   update converges at ≈5000 iterations (cell values become independent of the random
   start, verified over 5 inits, spread <0.5%); at the paper's 500 iterations cells are
   start-dominated (linear M=1: 2.07 → 0.10 converged, `verify_claims.py`). Our tables
   report the converged runs, the paper's numbers its 500-iteration budget; this is why
   some of our cells beat the paper's.
4. **Solvers.** The paper's MLP/SVM use Levenberg–Marquardt (MATLAB); ours use sklearn
   (L-BFGS / libsvm defaults), best-on-test selection as in the paper. On the small,
   high-dimensional real training sets this lowers our RMSE — the source of the
   remaining quantitative gaps.
5. **Orange Juice overflow.** The raw printed rule grows ‖W‖ by 1 + µ(1 − λ_min(Σ)) per
   iteration (measured 1.019–1.020); on the rank-deficient Orange Juice covariance
   (λ_min ≈ 0) it overflows double precision within the published budget. Sec. 4.2.2
   prescribes regularization for exactly this experiment: `src/extract.py::icafx_sec422`
   applies the published update (µ1 = µ2 = 0.02, 500 iterations, stop 1e-6) with the
   growth confined to the free scale of u — unit-norm row gauge (Eq. 6; direction
   untouched) and trust region |v_a| ≤ 1.2. These four cells are the only ones not
   computed by the unregularized published rule.
6. **Not specified in the paper** (assumptions): SIR slices (10), k-NN kernel exponent,
   sphering regularization, MLP init, seeds (42; all runs reproducible via `SEED` in
   `run_all.py`).
