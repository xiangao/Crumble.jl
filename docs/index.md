# Crumble.jl Documentation

## Overview

Crumble.jl is a Julia package for flexible and general causal mediation analysis using Riesz representers and deep learning. It implements a unified estimation strategy from Liu, Williams, Rudolph, and Diaz (2024).

## Tutorials

- [Getting Started](getting-started.md) — Basic tutorial
- [Main Vignette](vignette.md) — Comprehensive documentation

## API Reference

- [crumble()](crumble.md) — Main estimation function
- [crumble_control()](crumble_control.md) — Control parameters
- [sequential_module()](sequential_module.md) — Neural network architecture

## Effect Types

| Type | Code | Description |
|------|------|-------------|
| Natural | "N" | Natural direct/indirect effects |
| Organic | "O" | Organic effects |
| Randomized Interventional | "RI" | Randomized interventional effects |
| Recanting Twins | "RT" | Full decomposition (6 paths) |

## Installation

```julia
using Pkg
Pkg.add("Crumble")
```

Or for development:
```julia
Pkg.develop(path="/path/to/Crumble.jl")
```
