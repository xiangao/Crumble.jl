# SequentialModule

## Description

A function factory that creates neural network architectures for Riesz representer estimation.

## Usage

```julia
sequential_module(; layers::Int=1, hidden::Int=20, dropout::Float64=0.1)
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `layers` | 1 | Number of hidden layers |
| `hidden` | 20 | Number of hidden units |
| `dropout` | 0.1 | Dropout rate |

## Value

Returns a function that takes input dimension and returns a Flux Chain.

## Examples

```julia
# Default architecture
nn = sequential_module()

# Custom architecture
nn = sequential_module(layers = 2, hidden = 64, dropout = 0.2)

# Use with crumble
result = crumble(
    data = data,
    trt = ["A"],
    outcome = "Y",
    mediators = ["M"],
    covar = ["W1"],
    nn_module = nn,
    control = crumble_control(epochs = 50)
)
```

## Architecture Details

The default architecture:
- Input layer: `d_in → hidden` with ELU activation
- Hidden layers: `hidden → hidden` with ELU (if `layers > 0`)
- Output layer: `hidden → 1` with Dropout and Softplus
