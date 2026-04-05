# Crumble

## Description

Flexible and general causal mediation analysis using Riesz representers and deep learning.

## Usage

```julia
crumble(
    data::DataFrame;
    trt::Vector{String},
    outcome::String,
    mediators::Vector{String},
    moc::Union{Vector{String}, Nothing}=nothing,
    covar::Vector{String},
    obs::Union{String, Nothing}=nothing,
    id::Union{String, Nothing}=nothing,
    d0::Union{Function, Nothing}=nothing,
    d1::Union{Function, Nothing}=nothing,
    effect::String="RT",
    weights::Vector{Float64}=ones(nrow(data)),
    learners::Vector{String}=["glm"],
    nn_module=sequential_module(),
    control::CrumbleControl=crumble_control()
)
```

## Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `data` | `DataFrame` | Data in wide format |
| `trt` | `Vector{String}` | Treatment variable names |
| `outcome` | `String` | Outcome variable name |
| `mediators` | `Vector{String}` | Mediator variable names |
| `moc` | `Union{Vector{String}, Nothing}` | Mediator-outcome confounder names (required for RT, RI) |
| `covar` | `Vector{String}` | Baseline covariate names |
| `obs` | `Union{String, Nothing}` | Outcome observation indicator (0/1) |
| `id` | `Union{String, Nothing}` | Cluster ID |
| `d0` | `Union{Function, Nothing}` | Shift function for control |
| `d1` | `Union{Function, Nothing}` | Shift function for treatment |
| `effect` | `String` | Effect type: "N", "O", "RI", or "RT" |
| `weights` | `Vector{Float64}` | Survey weights |
| `learners` | `Vector{String}` | MLJ learner algorithms |
| `nn_module` | `Function` | Neural network architecture |
| `control` | `CrumbleControl` | Control parameters |

## Value

Returns a `CrumbleResult` with:

- `estimates` — Dictionary of estimates with SE, CI, p-values
- `outcome_reg` — Outcome regression predictions
- `alpha_n` — Natural density ratio estimates
- `alpha_r` — Randomized density ratio estimates
- `fits` — Fitted values
- `effect` — Effect type

## Examples

```julia
using Crumble
using DataFrames

data = DataFrame(
    A = rand([0, 1], 100),
    Y = rand(100),
    M = rand([0, 1], 100),
    W1 = randn(100),
)

result = crumble(
    data = data,
    trt = ["A"],
    outcome = "Y",
    mediators = ["M"],
    covar = ["W1"],
    effect = "N"
)

print(result)
```
