# Validation

`dgp.jl` — the `medoutcon` vignette DGP and its exact truth by enumeration.
`true_alpha.jl` — exact sequential Riesz representers (shows the weak-overlap cell).
`mc.jl` — Monte Carlo: `julia -t 6 --project=validation validation/mc.jl RI 1000 200 glm nn 100`
(effect, n, reps, learner, riesz, alpha_cap). Results in `results/`.
`xcheck_crumble.jl` / `xcheck_medoutcon.R` — the same n = 5000 sample through both packages.
`seed_var.jl` — spread across seeds on one sample (variance the IF SE treats as fixed).
