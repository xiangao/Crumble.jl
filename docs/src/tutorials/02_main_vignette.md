# Interventional effects with a known truth

```@meta
CurrentModule = Crumble
```

We use the data-generating process of the `medoutcon` vignette. Three binary
baseline covariates `W`, a binary treatment `A`, a binary mediator-outcome
confounder `Z` that `A` affects, a binary mediator `M` and a binary outcome `Y`.
Because `Z` is affected by `A` and confounds `M` and `Y`, natural effects are not
identified; randomized interventional effects are.

```@example vignette
using Crumble, DataFrames, Random

lg(x) = 1 / (1 + exp(-x))
pw3(w1, w2) = min(0.2 + (w1 + w2) / 3, 1.0)
pa(w1, w2, w3) = lg(w1 + w2 + w3 - 2)
pz(a, w1, w2, w3) = lg(-log(2) - a + (w1 + w2 + w3) / 3 + 0.2)
pm(a, z, w1, w2) = lg(log(3) * (w1 + w2) + 2a - 2z)
py(a, z, m, w1, w2, w3) = lg(1 / (w1 + w2 + w3 - z + a + m))

function simulate(n)
    w1 = rand(n) .< 0.6; w2 = rand(n) .< 0.3; w3 = rand(n) .< pw3.(w1, w2)
    a = rand(n) .< pa.(w1, w2, w3); z = rand(n) .< pz.(a, w1, w2, w3)
    m = rand(n) .< pm.(a, z, w1, w2); y = rand(n) .< py.(a, z, m, w1, w2, w3)
    DataFrame(W1 = Int.(w1), W2 = Int.(w2), W3 = Int.(w3), A = Int.(a), Z = Int.(z),
              M = Int.(m), Y = Int.(y))
end
nothing # hide
```

Every variable is binary, so each functional can be computed exactly. The
randomized functional behind the interventional direct effect is
``\theta(a', a^*) = E_W \sum_{z'} p(z' \mid a', W) \sum_{z,m} p(z \mid a^*, W)\, p(m \mid a^*, W, z)\, \mu(a', W, z', m)``,
and ``IDE = \theta(1, 0) - \theta(0, 0)``, ``IIE = \theta(1, 1) - \theta(1, 0)``.

```@example vignette
b(p, x) = x == 1 ? p : 1 - p
function theta(ap, as)
    s = 0.0
    for w1 in 0:1, w2 in 0:1, w3 in 0:1, zp in 0:1, z in 0:1, m in 0:1
        s += b(0.6, w1) * b(0.3, w2) * b(pw3(w1, w2), w3) * b(pz(ap, w1, w2, w3), zp) *
             b(pz(as, w1, w2, w3), z) * b(pm(as, z, w1, w2), m) * py(ap, zp, m, w1, w2, w3)
    end
    s
end
(IDE = theta(1, 0) - theta(0, 0), IIE = theta(1, 1) - theta(1, 0))
```

Now the estimate. The closed-form saturated Riesz representer is exact for
discrete data and fast; the neural-network default (`riesz = :nn`) is what the R
package uses and applies to continuous covariates too.

```@example vignette
Random.seed!(2026)
df = simulate(5000)
result = crumble(df, ["A"]; outcome = "Y", mediators = ["M"], moc = ["Z"],
                 covar = ["W1", "W2", "W3"], effect = "RI",
                 control = crumble_control(crossfit_folds = 5, riesz = :linear,
                                           riesz_basis = :saturated))
Crumble.tidy(result)
```

`ride` and `riie` are the interventional direct and indirect effects. The
standard errors come from the influence function of each contrast, so they account
for the two functionals being estimated on the same sample.
