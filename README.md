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
every functional can be computed by enumeration (`validation/`). With the neural
Riesz representers, `glm` outcome regressions, 5 folds and `alpha_cap = 100`:

| effect | n | reps | truth | bias | sd | mean se | 95% coverage |
|---|---|---|---|---|---|---|---|
| RI direct | 1000 | 200 | -0.0740 | 0.0066 | 0.0884 | 0.0849 | 0.915 |
| RI indirect | 1000 | 200 | -0.0246 | 0.0076 | 0.0668 | 0.0661 | 0.955 |
| RI direct | 5000 | 100 | -0.0740 | 0.0039 | 0.0281 | 0.0247 | 0.890 |
| RI indirect | 5000 | 100 | -0.0246 | 0.0039 | 0.0163 | 0.0174 | 0.930 |
| RT direct | 1000 | 200 | -0.0645 | 0.0049 | 0.0457 | 0.0378 | 0.930 |
| RT indirect | 1000 | 200 | -0.0252 | -0.0026 | 0.0175 | 0.0147 | 0.900 |
| RT ATE | 1000 | 200 | -0.0834 | -0.0031 | 0.0384 | 0.0336 | 0.895 |

The point estimates are close to unbiased; the sampling sd at n = 5000 (0.028)
matches the efficient standard error `medoutcon` reports for this design. Two
shortcomings remain. First, **the standard errors are mostly too small**, by up to 20%
(most of the RT paths, the RI direct effect at n = 5000), and coverage is 0.88-0.95
rather than 0.95. Part of the gap is variability the influence-function
SE treats as fixed — the Z' permutation, the folds and the network initialisation
(across-seed sd 0.005 on one n = 5000 sample for the RI direct effect) — but most
of it is not; with the closed-form saturated representers the SE is calibrated
(0.067 against sd 0.069 at n = 1000), which points at the regularised network
representers. Second, the recanting-twin path `p2` (A -> Z -> Y) recovers only
about half of its true value (-0.0082 against -0.0165). On identical data (n = 5000)
the RI direct effect is -0.058 (se 0.022) here and -0.070 (se 0.028) from
`medoutcon`.

Three departures from the R package, each deliberate:

- **Mini-batch weights.** In the Riesz loss the R code multiplies the full-sample
  weight vector by a mini-batch of outputs, which broadcasts to an `n x batch`
  matrix and so weights every observation by the mean weight. Here each batch uses
  its own observations' weights.
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
several thousand, and 107 of 200 replications missed the truth by more than 0.3. Check the fitted
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
