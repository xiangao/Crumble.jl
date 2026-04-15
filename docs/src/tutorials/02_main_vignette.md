# Crumble.jl Main Vignette

```@meta
CurrentModule = Crumble
```

`Crumble.jl` supports several mediation estimands:

- Natural direct and indirect effects
- Organic direct and indirect effects
- Randomized interventional effects
- Recanting twins decompositions

## Basic Usage

```@setup crumble_vignette
using Crumble
using DataFrames
using Random

Random.seed!(321)

n = 120
data = DataFrame(
    A = rand([0, 1], n),
    Y = rand(n),
    M = rand([0, 1], n),
    Z = rand([0, 1], n),
    W1 = randn(n),
    W2 = randn(n),
)

d0 = (data, trt) -> fill(0, nrow(data))
d1 = (data, trt) -> fill(1, nrow(data))
```

```@example crumble_vignette
result = crumble(
    data,
    ["A"],
    outcome = "Y",
    mediators = ["M"],
    covar = ["W1", "W2"],
    d0 = d0,
    d1 = d1,
    effect = "N",
    control = crumble_control(crossfit_folds = 2, epochs = 1, batch_size = 32)
)

Crumble.tidy(result)
```

## Recanting Twins

```@example crumble_vignette
result = crumble(
    data,
    ["A"],
    outcome = "Y",
    mediators = ["M"],
    moc = ["Z"],
    covar = ["W1", "W2"],
    d0 = d0,
    d1 = d1,
    effect = "RT",
    control = crumble_control(crossfit_folds = 2, epochs = 1, batch_size = 32)
)

print(result)
```

## Custom Neural Networks

```@example crumble_vignette
custom_nn = sequential_module(layers = 2, hidden = 32, dropout = 0.2)
custom_nn(3)
```
