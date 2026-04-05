# CrumbleControl

## Description

Create control parameters for the Crumble estimation procedure.

## Usage

```julia
crumble_control(;
    crossfit_folds::Int=10,
    mlr3superlearner_folds::Int=10,
    zprime_folds::Int=1,
    epochs::Int=100,
    learning_rate::Float64=0.01,
    batch_size::Int=64,
    device::String="cpu"
)
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `crossfit_folds` | 10 | Number of cross-fitting folds |
| `mlr3superlearner_folds` | 10 | Folds for MLJ Super Learner |
| `zprime_folds` | 1 | Folds for Z' permutation |
| `epochs` | 100 | Neural network training epochs |
| `learning_rate` | 0.01 | Learning rate for NN |
| `batch_size` | 64 | Mini-batch size |
| `device` | "cpu" | Device: "cpu", "cuda", or "mps" |

## Value

Returns a `CrumbleControl` object.

## Examples

```julia
# Fast settings for testing
control = crumble_control(
    crossfit_folds = 2,
    epochs = 5,
    batch_size = 32
)

# Production settings
control = crumble_control(
    crossfit_folds = 10,
    epochs = 100,
    learning_rate = 0.01,
    batch_size = 64,
    device = "cuda"
)
```
