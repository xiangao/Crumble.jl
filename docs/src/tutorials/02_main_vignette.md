# Crumble.jl Main Vignette

`Crumble.jl` supports several mediation estimands:

- Natural direct and indirect effects
- Organic direct and indirect effects
- Randomized interventional effects
- Recanting twins decompositions

## Basic Usage

```julia
using Crumble
using DataFrames

data = DataFrame(
    A = rand([0, 1], 500),
    Y = rand(500),
    M = rand([0, 1], 500),
    Z = rand([0, 1], 500),
    W1 = randn(500),
    W2 = randn(500),
)

d0 = (data, trt) -> fill(0, nrow(data))
d1 = (data, trt) -> fill(1, nrow(data))

result = crumble(
    data,
    ["A"],
    outcome = "Y",
    mediators = ["M"],
    covar = ["W1", "W2"],
    d0 = d0,
    d1 = d1,
    effect = "N",
    control = crumble_control(crossfit_folds = 3, epochs = 10)
)
```

## Recanting Twins

```julia
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
    control = crumble_control(crossfit_folds = 5, epochs = 50)
)
```

## Custom Neural Networks

```julia
custom_nn = sequential_module(layers = 2, hidden = 32, dropout = 0.2)

result = crumble(
    data,
    ["A"],
    outcome = "Y",
    mediators = ["M"],
    covar = ["W1", "W2"],
    nn_module = custom_nn,
    control = crumble_control(epochs = 100)
)
```
