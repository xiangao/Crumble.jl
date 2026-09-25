# Crumble.jl

> **⚠ Prototype — not usable for scientific inference.**
>
> A 2026-09-25 audit found that the core estimator was not implemented. The Riesz
> representers were filled with random numbers, a missing-alpha branch reported
> `mean(Y)` as the causal functional, a degenerate standard error was replaced by
> the constant `0.05`, contrast standard errors dropped the covariance between
> influence functions computed on the same sample, and the organic,
> randomized-interventional and randomized-transported routes all returned the
> natural-effect numbers under different labels.
>
> Those paths now raise explicit errors instead of returning fabricated values, so
> the package will refuse to run rather than mislead. The contrast standard error
> has been corrected to use the observation-level contrast influence function,
> `sd(IF_a - IF_b) / sqrt(n)`.
>
> **Do not use any number this package produced before 2026-09-25.** The estimating
> equations must be implemented and validated against a known data-generating
> process before release.


`Crumble.jl` is a Julia implementation of causal mediation estimators based on
Riesz representers. I use it for examples where the mediation target is written
as a functional and the nuisance functions are fit with neural networks.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/xiangao/Crumble.jl")
```

## Tutorials

Full documentation: **https://xiangao.github.io/Crumble.jl/**

| Tutorial | Description |
|----------|-------------|
| [Getting Started](https://xiangao.github.io/Crumble.jl/tutorials/01_getting_started/) | Package overview and basic mediation workflow |
| [Main Vignette](https://xiangao.github.io/Crumble.jl/tutorials/02_main_vignette/) | End-to-end example and core estimation pattern |
