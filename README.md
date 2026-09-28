# Crumble.jl

`Crumble.jl` estimates natural, organic, randomized interventional and
recanting-twin mediation effects. It is a Julia port of the R package
[`crumble`](https://cran.r-project.org/package=crumble) (Williams): every effect is a
contrast of functionals, each estimated by a cross-fitted one-step estimator that
combines sequential outcome regressions with sequentially fitted Riesz
representers.

## Status

Rebuilt on 2026-09-28. Versions before that date did not implement the estimator —
the Riesz representers were random numbers and the outcome regressions ignored the
treatment shifts — and **no number they produced should be used.**

The current version is checked against the exact truth of an all-binary DGP with a
treatment-affected mediator-outcome confounder (the `medoutcon` vignette's), where
every functional can be computed by enumeration (`validation/`).

**The influence functions are correct.** With every nuisance replaced by its true
value (`validation/oracle.jl`), all estimands are unbiased and the IF standard error
matches the sampling sd: coverage 0.944-0.957 at n = 1000 (1000 replications) and
0.93-0.96 at n = 5000 (500). The cross-unit dependence created by the Z' permutation
is negligible.

**The nuisance fits decide the rest.** With nonparametric fits (`learners =
["saturated"]`, `riesz = :linear, riesz_basis = :saturated`) the estimator is
unbiased with coverage 0.94-0.955 at n = 1000, recanting-twin paths included:

| effect | n | truth | bias | sd | mean se | coverage |
|---|---|---|---|---|---|---|
| RT direct | 1000 | -0.0645 | -0.0002 | 0.0508 | 0.0508 | 0.940 |
| RT indirect | 1000 | -0.0252 | -0.0029 | 0.0317 | 0.0325 | 0.955 |
| RT ATE | 1000 | -0.0834 | -0.0009 | 0.0379 | 0.0381 | 0.945 |
| RT p2 (A->Z->Y) | 1000 | -0.0165 | 0.0008 | 0.0214 | 0.0229 | 0.950 |

**Network representers: the R defaults are not used.** R crumble's network uses
dropout 0.1 and weight decay 0.01. Both shrink the fitted representers toward zero:
on an n = 5000 sample their means were 0.81-0.87 where they must be 1, with a slope
of 0.66-0.89 on the true representer. The one-step correction is then only partly
applied, which left bias (the RT path p2 recovered half its value) and standard
errors up to 20% too small (coverage 0.88-0.95; `validation/results/*_Rdefaults.csv`).
The defaults here are dropout 0 and weight decay 0, which on the same sample give
means 1.005-1.013 and slopes 0.96-1.07, as good as the exact saturated fit
(`validation/alpha_quality.jl`). A Monte Carlo run of the estimator with these
defaults has not yet been completed; until it is, the nonparametric fits above are
the validated configuration for discrete data.

Four departures from the R package, each deliberate:

- **Mini-batch weights.** In the Riesz loss the R code multiplies the full-sample
  weight vector by a mini-batch of outputs, which broadcasts to an `n x batch`
  matrix and so weights every observation by the mean weight. Here each batch uses
  its own observations' weights.
- **Network regularisation.** Dropout 0 and weight decay 0 by default, for the
  reason above; `sequential_module(dropout = 0.1)` and
  `crumble_control(weight_decay = 0.01)` restore R's settings.
- **Default shifts.** With a single 0/1 treatment, omitted `d0`/`d1` default to
  setting it to 0 and 1. (In R an omitted shift leaves the treatment unchanged and
  every effect is zero.)
- **Learners.** One outcome learner is fitted, `"glm"` (the R default) or
  `"saturated"` (cell means, nonparametric for discrete regressors); there is no
  super learner.

It adds a closed-form Riesz option (`riesz = :linear`, with `riesz_basis = :saturated`
the exact empirical representer for discrete data) and a representer bound
(`alpha_cap`), for which see the next section.

## Weak overlap

The Riesz loss penalises a representer only at observed covariate patterns but
rewards it at shifted ones. When a training fold contains no observation in a
pattern the shift needs, the loss has no minimiser there and the neural network
drifts to arbitrarily large values. In the validation DGP at n = 1000 the true
fourth-stage representer of the interventional direct effect peaks at 45, in a cell
with about two expected observations per training fold; unbounded fits reached
several thousand, and with R's network settings 107 of 200 replications missed the
truth by more than 0.3. Check the fitted
representers (`result.alpha_r`, `result.alpha_n`) and bound them with `alpha_cap`
when overlap is weak.

## Usage

```julia
using Crumble, DataFrames

result = crumble(df, ["A"];
                 outcome = "Y", mediators = ["M"], moc = ["Z"],
                 covar = ["W1", "W2", "W3"], effect = "RI",
                 control = crumble_control(alpha_cap = 100.0))
result.estimates["ride"]      # randomized interventional direct effect
```

`effect` is `"N"`, `"O"`, `"RI"` or `"RT"`; `"RI"` and `"RT"` need `moc`.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/xiangao/Crumble.jl")
```

Documentation: https://xiangao.github.io/Crumble.jl/
