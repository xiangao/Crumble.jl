# Getting started: natural effects

```@meta
CurrentModule = Crumble
```

A binary covariate `W`, treatment `A`, mediator `M` and outcome `Y`, with no
mediator-outcome confounder affected by treatment, so natural direct and indirect
effects are identified.

```@example gs
using Crumble, DataFrames, Random

lg(x) = 1 / (1 + exp(-x))
pa(w) = lg(-0.5 + w); pm(a, w) = lg(-1 + 1.5a + 0.5w); py(a, m, w) = lg(-1 + 0.7a + 1.2m + 0.4w)

Random.seed!(1)
n = 5000
W = Int.(rand(n) .< 0.5); A = Int.(rand(n) .< pa.(W))
M = Int.(rand(n) .< pm.(A, W)); Y = Int.(rand(n) .< py.(A, M, W))
df = DataFrame(W = W, A = A, M = M, Y = Y)
nothing # hide
```

The truth: ``E[Y(a, M(a^*))] = \sum_w p(w) \sum_m p(m \mid a^*, w)\, \mu(a, m, w)``.

```@example gs
th(a, as) = sum(0.5 * sum((m == 1 ? pm(as, w) : 1 - pm(as, w)) * py(a, m, w) for m in 0:1) for w in 0:1)
(direct = th(1, 0) - th(0, 0), indirect = th(1, 1) - th(1, 0), ate = th(1, 1) - th(0, 0))
```

```@example gs
result = crumble(df, ["A"]; outcome = "Y", mediators = ["M"], covar = ["W"], effect = "N",
                 control = crumble_control(crossfit_folds = 5, riesz = :linear,
                                           riesz_basis = :saturated))
print(result)
```

With a single 0/1 treatment the shifts default to setting it to 0 and 1; pass
`d0` and `d1` — functions `(data, trt) -> new values` — for any other contrast.
`Crumble.tidy(result)` returns the same numbers as a `DataFrame`.
