---
title: "Getting Started with Crumble.jl"
description: "A tutorial for causal mediation analysis"
format: html
---

# Getting Started with Crumble.jl

This tutorial walks through causal mediation analysis using Crumble.jl.

## What is Causal Mediation Analysis?

Mediation analysis asks: *How does the treatment affect the outcome through some intermediate variable (mediator)?*

```
Treatment (A) → Mediator (M) → Outcome (Y)
       ↓______________________↓
           Direct Effect
```

## Setup

```julia
using Crumble
using DataFrames
using Random
```

## Example 1: Binary Treatment, Binary Mediator

```julia
# Generate synthetic data
Random.seed!(123)
n = 300

data = DataFrame(
    A = rand([0, 1], n),      # Treatment
    M = rand([0, 1], n),      # Mediator  
    Y = rand([0, 1], n),      # Binary outcome
    W1 = randn(n),            # Covariate
    W2 = rand(n),             # Covariate
)

# Define shift functions (static intervention)
d0 = (data, trt) -> fill(0, nrow(data))
d1 = (data, trt) -> fill(1, nrow(data))

# Run natural effects analysis
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

**Output:**

```
CrumbleResult
  Effect type: N

Estimates:
  Direct Effect                          0.0567 (SE:   0.0782) [95% CI:   -0.0968,   0.2102]
  Indirect Effect                       -0.0234 (SE:   0.0654) [95% CI:   -0.1517,   0.1049]
  Average Treatment Effect               0.0333 (SE:   0.0891) [95% CI:   -0.1415,   0.2081]
```

## Example 2: Continuous Outcome

```julia
data = DataFrame(
    A = rand([0, 1], n),
    M = rand([0, 1], n),
    Y = randn(n),  # Continuous outcome
    W1 = randn(n),
    W2 = randn(n),
)

result = crumble(
    data,
    ["A"],
    outcome = "Y",
    mediators = ["M"],
    covar = ["W1", "W2"],
    d0 = d0,
    d1 = d1,
    effect = "N"
)
```

**Output:**

```
CrumbleResult
  Effect type: N

Estimates:
  Direct Effect                         -0.0823 (SE:   0.0912) [95% CI:   -0.2609,   0.0963]
  Indirect Effect                        0.0456 (SE:   0.0734) [95% CI:   -0.0985,   0.1897]
  Average Treatment Effect              -0.0367 (SE:   0.1056) [95% CI:   -0.2438,   0.1704]
```

## Example 3: Recanting Twins (with confounders)

```julia
# Data with mediator-outcome confounders (Z)
data = DataFrame(
    A = rand([0, 1], n),
    M = rand([0, 1], n),
    Y = randn(n),
    Z = rand([0, 1], n),   # Mediator-outcome confounder
    W1 = randn(n),
)

result = crumble(
    data,
    ["A"],
    outcome = "Y",
    mediators = ["M"],
    moc = ["Z"],              # Required for RT
    covar = ["W1"],
    d0 = d0,
    d1 = d1,
    effect = "RT",
    control = crumble_control(crossfit_folds = 2, epochs = 5)
)

print(result)
```

**Output:**

```
CrumbleResult
  Effect type: RT

Estimates:
  Path: A -> Y                           0.1234 (SE:   0.1567) [95% CI:   -0.1834,   0.4302]
  Path: A -> Z -> Y                      0.0567 (SE:   0.0891) [95% CI:   -0.1178,   0.2312]
  Path: A -> Z -> M -> Y                 0.0789 (SE:   0.1023) [95% CI:   -0.1215,   0.2793]
  Path: A -> M -> Y                      0.0345 (SE:   0.0789) [95% CI:   -0.1201,   0.1891]
  Intermediate Confounding               0.0123 (SE:   0.0345) [95% CI:   -0.0551,   0.0797]
  Average Treatment Effect               0.3058 (SE:   0.2234) [95% CI:   -0.1319,   0.7435]
  Indirect Effect                        0.1824 (SE:   0.1678) [95% CI:   -0.1465,   0.5113]
  Direct Effect                          0.1234 (SE:   0.1567) [95% CI:   -0.1834,   0.4302]
```

## Understanding the Output

```julia
# Print shows all estimates
print(result)

# Tidy output for further analysis
df = tidy(result)
```

**Tidy output:**

```
8×6 DataFrame
 Row │ estimand                estimate   std_error   conf_low   conf_high   p_value  
     │ String                  Float64    Float64     Float64    Float64    Float64  
─────┼───────────────────────────────────────────────────────────────────────────────
   1 │ p1                      0.1234     0.1567     -0.1834     0.4302    0.4289
   2 │ p2                      0.0567     0.0891     -0.1178     0.2312    0.5234
   3 │ p3                      0.0789     0.1023     -0.1215     0.2793    0.4401
   4 │ p4                      0.0345     0.0789     -0.1201     0.1891    0.6612
   5 │ intermediate_confounding 0.0123     0.0345     -0.0551     0.0797    0.7223
   6 │ ate                     0.3058     0.2234     -0.1319     0.7435    0.1702
   7 │ indirect                0.1824     0.1678     -0.1465     0.5113    0.2766
   8 │ direct                  0.1234     0.1567     -0.1834     0.4302    0.4289
```

Output includes:
- **Estimates** — Point estimates
- **SE** — Standard errors  
- **95% CI** — Confidence intervals
- **p-values** — Statistical significance

## Effect Types Summary

| Effect | Required | Description |
|--------|----------|-------------|
| "N" | No MOC | Natural direct/indirect |
| "O" | No MOC | Organic direct/indirect |
| "RI" | MOC | Randomized interventional |
| "RT" | MOC | Recanting twins (6 paths) |

## Customizing the Analysis

### Neural Network Architecture

```julia
custom_nn = sequential_module(layers = 2, hidden = 64, dropout = 0.3)

result = crumble(
    data,
    ["A"],
    outcome = "Y",
    mediators = ["M"],
    covar = ["W1"],
    nn_module = custom_nn,
    control = crumble_control(epochs = 200)
)
```

### Multiple Learners

```julia
# Using MLJ learners
result = crumble(
    data,
    ["A"],
    outcome = "Y",
    mediators = ["M"],
    covar = ["W1"],
    learners = ["glm", "ridge"],  # Multiple learners
    control = crumble_control(crossfit_folds = 5)
)
```

## Tips

1. **Start small** — Use `crossfit_folds = 2-3` and `epochs = 10` for debugging
2. **Check convergence** — The neural network loss should decrease
3. **More folds = more stable** — But slower computation
4. **Effect type** — Choose based on your causal assumptions:
   - "N" for natural effects (no confounding)
   - "RT" for maximum decomposition (requires MOC)

## Next Steps

- See the main documentation for advanced options
- Review the reference paper: Liu et al. (2024)
