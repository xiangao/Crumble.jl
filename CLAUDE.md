# Crumble.jl — project notes for Claude

Julia port (from an R package) of causal mediation analysis via Riesz
representers / efficient influence functions, with neural-network (Flux)
nuisance estimation and cross-fitting. Supports natural, organic, and
randomized-interventional (in)direct effects, plus recanting-twins
decompositions when a mediator-outcome confounder (`moc`) affected by
treatment is present.

## Core API

- `crumble(data, trt; outcome, mediators, covar, d0, d1, effect, moc=..., control=...)`
  — main entry point. `effect` ∈ `"N"` (natural), `"O"` (organic),
  `"RT"` (recanting twins), `"RI"` (randomized interventional).
  `d0`/`d1` are shift functions `(data, trt) -> vector` defining the
  treatment contrast (not just 0/1 — see `src/shift.jl`).
- `crumble_control(; crossfit_folds=10, epochs=100, learning_rate=0.01,
  batch_size=64, device="cpu", ...)` — tunable knobs, returns `CrumbleControl`.
- `sequential_module(; layers=1, hidden=20, dropout=0.1)` — factory for the
  default Flux MLP used to fit nuisance functions (`alpha`/`theta`/`phi`).
- `Crumble.tidy(result)` — `CrumbleResult` → tidy `DataFrame`; `print(result)`
  gives a formatted summary.

`effect = "RT"` or `"RI"` requires `moc` to be set; `effect = "N"` or `"O"`
requires it to be `nothing` — enforced by `assert_effect_type` (see
`src/assertions.jl`), which throws `ArgumentError` on mismatch.

## What's where

- `src/main.jl` — `crumble()` orchestration: builds `CrumbleVars`/`CrumbleData`,
  cross-fit folds, dispatches to `theta`/`phi`/`eif`/`calc_estimates` per effect.
- `src/types.jl` — `CrumbleVars`, `CrumbleData` (holds shifted copies
  `data_0`/`data_1`/`data_0zp`/`data_1zp`), `CrumbleControl`, `CrumbleResult`,
  `CrossFitFold`.
- `src/theta.jl`, `src/phi.jl`, `src/alpha.jl`, `src/eif.jl` — nuisance
  regressions (outcome/mediator models), Riesz representer (`alpha`) fits,
  and efficient-influence-function assembly per shift.
- `src/calc_estimates.jl` — `calc_estimates_natural/organic/rt/ri` combine
  per-arm EIFs (keyed by treatment-pattern strings like `"100"`/`"000"`/`"111"`)
  into direct/indirect/ATE point estimates + SEs.
- `src/shift.jl` — stochastic/static intervention shift functions (`d0`, `d1`).
- `src/dataset.jl`, `src/helpers.jl`, `src/assertions.jl`, `src/display.jl`,
  `src/params.jl`, `src/permutation.jl` — data prep (incl. one-hot encoding
  of `moc`), cross-fit fold construction, input validation, `print`/`tidy`.
- `test/runtests.jl` — unit tests on internals (control defaults, folds,
  weight normalization, `calc_estimates_natural` arithmetic, shift functions);
  no full end-to-end `crumble()` run (that's exercised via the doc examples).

## Tests

```
cd ~/projects/software/Crumble.jl && julia --project=. test/runtests.jl
```
24/24 pass, ~4s once precompiled (first run after a Julia/package update
pays ~2 min in MLJ-stack precompilation — that's normal, not a hang).

## Docs

Plain **Documenter.jl**, not Quarto — despite the README calling the pages
"Tutorials," `docs/src/*.md` are Documenter `@meta`/`@setup`/`@example`
blocks, not `.qmd`. `docs/make.jl` builds; `.github/workflows/docs.yml`
builds on every push/PR to `main` and deploys to GitHub Pages only on push
to `main`. Live at <https://xiangao.github.io/Crumble.jl/>. The README is
intentionally thin (installation + links) — `docs/src/index.md`,
`01_getting_started.md`, and `02_main_vignette.md` are the real source of
usage examples (including the `RT`/recanting-twins and custom-`sequential_module`
patterns) and are the first place to look when the README isn't enough.

## Gotchas

- `outcome` and `obs` columns must be binary 0/1 (`assert_binary_0_1`) —
  the vignettes use a binary `Y`, but `crumble` also accepts continuous `Y`
  in the main vignette's basic-usage example (only `outcome`, not a
  hardcoded set, is asserted binary — check `assert_binary_0_1` call sites
  in `main.jl` before assuming continuous outcomes are unsupported).
- `moc` (mediator-outcome confounder) columns get one-hot encoded and folded
  into `vars.Z` inside `CrumbleData` — if you're debugging fold/column
  mismatches, remember the effective `Z` symbol list is NOT what you passed
  in, it's the one-hot-expanded version.
- Cross-fit `folds` stratify on `outcome`, not on treatment — see the
  `strata` line in `crumble()`.

## 2026-09-28 — Estimator rebuilt (v0.2.0)

Before this, nothing was implemented: random Riesz representers (quarantined
2026-09-25), and `theta.jl` ignored the j/k/l shifts, fitted OLS without an
intercept, and stored training predictions where the EIF needs validation ones. The
Z' assignment omitted R's trace(P) = 0 constraint, so Z' = Z. Rewritten from R
crumble 0.1.2: `learners.jl` (glm | saturated), `riesz.jl` (NN port with per-batch
weights — R broadcasts the full weight vector against the batch — or closed-form
:linear/:saturated), `theta.jl`, `eif.jl` (IFEstimate, cluster-robust SE),
`calc_estimates.jl` (N/O/RI/RT), `permutation.jl` (Hungarian, diagonal priced out).
`alpha_cap` added: at n = 1000 an empty cell in a fold sends the NN representer to
thousands (true max 45). Validation in `validation/` (exact truth by enumeration);
results and the two open problems (SEs up to 20% low; RT path p2 half-recovered)
are in the README. 94 unit tests pass. Deps pruned (MLJ, JuMP, HiGHS gone).

## 2026-09-28 (later) — Root cause of the SE under-coverage and the p2 bias

`validation/oracle.jl` (true nuisances plugged into the EIFs) is unbiased and
calibrated everywhere, so the EIFs, the SE and the Z' permutation are right; the
problem is the nuisance fits. `validation/alpha_quality.jl` found it: R crumble's
network defaults (dropout 0.1, weight decay 0.01) shrink the fitted representers
(mean 0.81-0.87 where it must be 1). Defaults are now dropout 0 and weight decay 0
(`weight_decay` added to `crumble_control`), which fit as well as the exact
saturated representers. A full Monte Carlo with the new defaults was started and
stopped at xao's request after ~30 min; it has NOT been completed. Nonparametric
fits (saturated learner + saturated Riesz) are validated (coverage 0.94-0.955 at
n = 1000). A CV-ridge interaction basis was tried and reverted: it blew up in
sparse cells. Old-default MC results are kept as `*_Rdefaults.csv`.
