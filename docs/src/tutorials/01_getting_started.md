# Getting Started with Crumble.jl

```@meta
CurrentModule = Crumble
```

`Crumble.jl` targets mediation problems where direct and indirect effects may be estimated flexibly with machine learning.

This page keeps the examples lightweight for documentation purposes. The code is shown in the same structure you would use in real analyses.

## Setup

```@setup crumble_getting_started
using Crumble
using DataFrames
using Random

Random.seed!(123)
n = 120

data = DataFrame(
    A = rand([0, 1], n),
    M = rand([0, 1], n),
    Y = rand([0, 1], n),
    W1 = randn(n),
    W2 = rand(n),
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
    control = crumble_control(crossfit_folds = 2, epochs = 1, batch_size = 32)
)
```

## Example: Binary Treatment and Binary Mediator

```@example crumble_getting_started
print(result)
```

## Tidy Output

```@example crumble_getting_started
Crumble.tidy(result)
```

Use `print(result)` for the formatted summary and `tidy(result)` when you want a `DataFrame` for downstream reporting.
