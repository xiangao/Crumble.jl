# Crumble.jl

`Crumble.jl` estimates natural, organic, randomized interventional and
recanting-twin mediation effects, following the R package `crumble`. Each effect is
a contrast of functionals estimated by a cross-fitted one-step estimator built from
sequential outcome regressions and sequentially fitted Riesz representers.

Versions before 2026-09-28 did not implement the estimator; see the README.

## Tutorials

- [Getting started: natural effects](tutorials/01_getting_started.md)
- [Interventional effects with a known truth](tutorials/02_main_vignette.md)

## Core API

- `crumble`
- `crumble_control`
- `sequential_module`
