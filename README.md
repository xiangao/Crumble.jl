# Crumble.jl

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
