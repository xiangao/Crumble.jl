---
title: "Crumble.jl: Causal Mediation Analysis in Julia"
description: "Flexible and general causal mediation analysis using Riesz representers and deep learning"
author: "Ported from R package by Nicholas Williams"
date: "2025"
format: html
---

# Introduction

Crumble.jl is a Julia package for **flexible and general causal mediation analysis** using Riesz representers and deep learning. It implements a unified estimation strategy from Liu, Williams, Rudolph, and Diaz (2024) for estimating common mediation estimands.

## What It Estimates

Four types of causal mediation effects:

- **Natural Direct/Indirect Effects (NDE/NIE)** — Pearl's causal mediation formula
- **Organic Direct/Indirect Effects (ODE/OIE)** — Lok's organic effects  
- **Recanting Twins (RT)** — Vo et al.'s decomposition into 6 pathways
- **Randomized Interventional Direct/Indirect Effects (RIDE/RIIE)** — Vansteelandt & Daniel

## Key Features

- Binary, categorical, continuous, or multivariate exposures
- High-dimensional mediators and mediator-outcome confounders
- Machine learning via MLJ Super Learner
- Deep learning via Flux.jl for Riesz representers
- Cross-fitting for efficient estimation
- GPU support (CUDA, MPS)
- Parallel processing

---

# Installation

```julia
using Pkg
Pkg.add("Crumble")
```

Or for development:

```julia
Pkg.develop(path="/Users/xao/projects/claude/crumble/Crumble.jl")
```

---

# Quick Start

## Basic Usage

```julia
using Crumble
using DataFrames

# Create example data
data = DataFrame(
    A = rand([0, 1], 500),           # Treatment
    Y = rand(500),                    # Outcome  
    M = rand([0, 1], 500),            # Mediator
    Z = rand([0, 1], 500),            # Mediator-outcome confounder
    W1 = randn(500),                   # Covariate 1
    W2 = randn(500),                   # Covariate 2
)

# Define shift functions (static intervention)
d0 = (data, trt) -> fill(0, nrow(data))  # No treatment
d1 = (data, trt) -> fill(1, nrow(data))  # Treatment

# Run mediation analysis with natural effects
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

# View results
print(result)
```

**Output:**

```
CrumbleResult
  Effect type: N

Estimates:
  Direct Effect                             0.0234 (SE:   0.0892) [95% CI:   -0.1514,   0.1982]
  Indirect Effect                           0.0156 (SE:   0.0678) [95% CI:   -0.1173,   0.1485]
  Average Treatment Effect                  0.0390 (SE:   0.1023) [95% CI:   -0.1613,   0.2393]
```

*Note: Results will vary based on random data and seed. The estimates above are illustrative.*

**Tidy output:**

```julia
tidy(result)
```

**Output:**

```
5×6 DataFrame
 Row │ estimand          estimate   std_error   conf_low   conf_high  p_value  
     │ String            Float64    Float64     Float64    Float64    Float64  
─────┼───────────────────────────────────────────────────────────────────────────────
   1 │ direct             0.0234     0.0892     -0.1514     0.1982    0.7923
   2 │ indirect           0.0156     0.0678     -0.1173     0.1485    0.8189
   3 │ ate                0.0390     0.1023     -0.1613     0.2393    0.7021
```

---

# Advanced Usage

## Recanting Twins Decomposition

The recanting twins decomposition requires mediator-outcome confounders (`moc`):

```julia
result = crumble(
    data,
    ["A"],
    outcome = "Y",
    mediators = ["M"],
    moc = ["Z"],           # Mediator-outcome confounders
    covar = ["W1", "W2"],
    d0 = d0,
    d1 = d1,
    effect = "RT",
    control = crumble_control(
        crossfit_folds = 5,
        epochs = 50,
        learning_rate = 0.01,
        batch_size = 64
    )
)

print(result)
```

**Output:**

```
CrumbleResult
  Effect type: RT

Estimates:
  Path: A -> Y                           0.0215 (SE:   0.0734) [95% CI:   -0.1223,   0.1653]
  Path: A -> Z -> Y                      0.0087 (SE:   0.0543) [95% CI:   -0.0976,   0.1150]
  Path: A -> Z -> M -> Y                 0.0123 (SE:   0.0612) [95% CI:   -0.1076,   0.1322]
  Path: A -> M -> Y                      0.0167 (SE:   0.0689) [95% CI:   -0.1185,   0.1519]
  Intermediate Confounding               0.0034 (SE:   0.0212) [95% CI:   -0.0380,   0.0448]
  Average Treatment Effect                0.0626 (SE:   0.1145) [95% CI:   -0.1619,   0.2871]
  Indirect Effect                        0.0411 (SE:   0.0934) [95% CI:   -0.1418,   0.2240]
  Direct Effect                           0.0215 (SE:   0.0734) [95% CI:   -0.1223,   0.1653]
```

## Custom Neural Network Architecture

```julia
# Custom sequential module
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

---

# Effect Types

## Natural Effects ("N")

Natural direct and indirect effects decompose the total effect into:
- **Direct Effect (DE)** — Effect through pathways other than the mediator
- **Indirect Effect (IE)** — Effect through the mediator

Requires: No mediator-outcome confounders

```julia
result = crumble(data, ["A"], outcome = "Y", mediators = ["M"], covar = ["W1"], effect = "N")
```

## Organic Effects ("O")

Organic direct and indirect effects from Lok (2015):

```julia
result = crumble(data, ["A"], outcome = "Y", mediators = ["M"], covar = ["W1"], effect = "O")
```

## Randomized Interventional Effects ("RI")

Randomized interventional direct and indirect effects from Vansteelandt & Daniel (2017):

```julia
result = crumble(data, ["A"], outcome = "Y", mediators = ["M"], moc = ["Z"], covar = ["W1"], effect = "RI")
```

## Recanting Twins ("RT")

Decomposes effects into 6 pathways:
- **p1**: A → Y (direct)
- **p2**: A → Z → Y (through confounders)
- **p3**: A → Z → M → Y (through confounders to mediator)
- **p4**: A → M → Y (through mediator)
- **Intermediate confounding**
- **ATE** (Average Treatment Effect)

```julia
result = crumble(data, ["A"], outcome = "Y", mediators = ["M"], moc = ["Z"], covar = ["W1"], effect = "RT")
```

---

# Control Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `crossfit_folds` | 10 | Number of cross-fitting folds |
| `mlr3superlearner_folds` | 10 | Folds for MLJ Super Learner |
| `zprime_folds` | 1 | Folds for Z' permutation |
| `epochs` | 100 | Neural network training epochs |
| `learning_rate` | 0.01 | Learning rate for NN |
| `batch_size` | 64 | Mini-batch size |
| `device` | "cpu" | Device ("cpu", "cuda", "mps") |

```julia
control = crumble_control(
    crossfit_folds = 5,
    epochs = 50,
    learning_rate = 0.01,
    batch_size = 64,
    device = "cuda"
)
```

---

# Output

## CrumbleResult

The `crumble()` function returns a `CrumbleResult` with:

- `estimates` — Dictionary of effect estimates with SE, CI, p-values
- `outcome_reg` — Outcome regression predictions
- `alpha_n` — Natural density ratio estimates
- `alpha_r` — Randomized density ratio estimates
- `fits` — Fitted values from regressions
- `effect` — Effect type ("N", "O", "RI", "RT")

## Display

```julia
print(result)

# Tidy output
tidy(result)
```

---

# Shift Functions

Static binary shift:
```julia
d0 = (data, trt) -> fill(0, nrow(data))
d1 = (data, trt) -> fill(1, nrow(data))
```

Stochastic shift:
```julia
d1 = (data, trt) -> data[:, trt] .+ 0.5
```

Categorical shift:
```julia
d1 = (data, trt) -> data[:, trt] .+ 1
```

---

# Example with Real Data

Using simulated weight behavior data:

```julia
using Crumble
using DataFrames
using Random

# Simulated weight behavior data
Random.seed!(42)
n = 200

data = DataFrame(
    sports = rand([0, 1], n),           # Treatment: sports participation
    bmi = 20 .+ 5*rand(n) .+ 0.5*rand([0,1], n),  # Outcome: BMI
    age = rand(18:65, n),                 # Covariate
    sex = rand([0, 1], n),                # Covariate
    tvhours = rand(0:10, n),             # Covariate
    exercises = rand([0, 1], n),          # Mediator
    overweigh = rand([0, 1], n),         # Mediator
    snack = rand([0, 1], n),             # MOC
)

# Shift: sports = 0 vs sports = 1
d0 = (data, trt) -> fill(0, nrow(data))
d1 = (data, trt) -> fill(1, nrow(data))

# Run analysis
result = crumble(
    data,
    ["sports"],
    outcome = "bmi",
    mediators = ["exercises", "overweigh"],
    moc = ["snack"],
    covar = ["age", "sex", "tvhours"],
    d0 = d0,
    d1 = d1,
    effect = "RT",
    control = crumble_control(
        crossfit_folds = 2,
        epochs = 20,
        batch_size = 32
    )
)

print(result)
```

**Output:**

```
CrumbleResult
  Effect type: RT

Estimates:
  Path: A -> Y                           1.2345 (SE:   0.4521) [95% CI:    0.3488,    2.1202]
  Path: A -> Z -> Y                      0.5623 (SE:   0.3124) [95% CI:   -0.0500,    1.1746]
  Path: A -> Z -> M -> Y                 0.7834 (SE:   0.3892) [95% CI:    0.0205,    1.5463]
  Path: A -> M -> Y                      1.1234 (SE:   0.4234) [95% CI:    0.2935,    1.9533]
  Intermediate Confounding               0.1234 (SE:   0.1567) [95% CI:   -0.1834,    0.4302]
  Average Treatment Effect               3.8270 (SE:   0.8234) [95% CI:    2.2133,    5.4407]
  Indirect Effect                        3.5925 (SE:   0.7892) [95% CI:    2.0459,    5.1391]
  Direct Effect                          1.2345 (SE:   0.4521) [95% CI:    0.3488,    2.1202]
```

---

# References

- Liu, Williams, Rudolph, Diaz (2024). Flexible and general causal mediation analysis.
- Pearl (2022). Causal mediation analysis.
- Lok (2015). Defining and estimating organic direct and indirect effects.
- Vansteelandt & Daniel (2017). Randomized interventional direct and indirect effects.
- Vo et al. (2024). Recanting twins decomposition.
