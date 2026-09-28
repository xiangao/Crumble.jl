# Fitted vs true Riesz representers on one sample: is the network shrinking them?
using Crumble, DataFrames, Random, Statistics, Printf, LinearAlgebra
include(joinpath(@__DIR__, "dgp.jl"))
include(joinpath(@__DIR__, "true_alpha_fns.jl"))
n = parse(Int, ARGS[1]); cfg = ARGS[2]
rng = Xoshiro(7); d = make_example_data(n; rng = rng)
ctrl = cfg == "nn" ? crumble_control(crossfit_folds = 5, alpha_cap = 100.0) :
       cfg == "nn_nowd" ? crumble_control(crossfit_folds = 5, alpha_cap = 100.0, weight_decay = 0.0) :
       crumble_control(crossfit_folds = 5, riesz = :linear, riesz_basis = :saturated)
mod = cfg == "nn_nowd" ? sequential_module(dropout = 0.0) : sequential_module()
Random.seed!(1)
r = crumble(d, ["A"]; outcome = "Y", mediators = ["M"], moc = ["Z"], covar = ["W_1", "W_2", "W_3"],
            effect = "RT", nn_module = mod, control = ctrl)
@printf "%-5s %-7s %8s %8s %8s %8s %8s\n" "key" "stage" "mean_f" "mean_t" "sd_f" "sd_t" "slope"
for (key, a) in sort(collect(r.alpha_r); by = first), s in ("alpha3", "alpha4")
    t = true_alpha_r(d, parse.(Int, collect(key))..., s)
    f = a[s]
    @printf "%-5s %-7s %8.3f %8.3f %8.3f %8.3f %8.3f\n" key s mean(f) mean(t) std(f) std(t) cov(f, t) / var(t)
end
