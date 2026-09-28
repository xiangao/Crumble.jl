# Monte Carlo validation of Crumble.jl against the exact truth of the book DGP.
# Run: julia -t 6 --project=validation validation/mc.jl <effect> <n> <reps> <learner> <riesz>
using Crumble, DataFrames, CSV, Random, Statistics, Printf, LinearAlgebra
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "dgp.jl"))

effect, n, reps, learner, rieszarg = ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3]), ARGS[4], Symbol(ARGS[5])
# riesz argument: nn | linear (saturated basis) | interactions (CV-ridge interaction basis)
riesz = rieszarg == :nn ? :nn : :linear
basis = rieszarg == :interactions ? :interactions : rieszarg == :linear ? :saturated : :main
cap = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : Inf
T = truth()
keys_ = effect == "RI" ? ["ride", "riie"] : ["direct", "indirect", "ate", "p1", "p2", "p3", "p4"]
tkey(k) = effect == "RI" ? "RI_" * k : "RT_" * k

rows = Vector{Any}(undef, reps)
Threads.@threads for r in 1:reps
    Random.seed!(10_000 + r)
    d = make_example_data(n)
    res = crumble(d, ["A"]; outcome = "Y", mediators = ["M"], moc = ["Z"], covar = ["W_1", "W_2", "W_3"],
                  effect = effect, learners = [learner],
                  control = crumble_control(crossfit_folds = 5, riesz = riesz,
                                            riesz_basis = basis, alpha_cap = cap))
    rows[r] = [(rep = r, estimand = k, estimate = res.estimates[k]["estimate"],
                se = res.estimates[k]["std.error"], truth = T[tkey(k)]) for k in keys_]
end
df = DataFrame(reduce(vcat, rows))
out = joinpath(@__DIR__, "results")
mkpath(out)
CSV.write(joinpath(out, "mc_$(effect)_n$(n)_$(learner)_$(rieszarg)_cap$(cap).csv"), df)

@printf "%s n=%d reps=%d learner=%s riesz=%s cap=%s\n" effect n reps learner rieszarg cap
@printf "%-10s %9s %9s %9s %9s %9s %9s\n" "estimand" "truth" "bias" "sd" "mean_se" "rmse" "cover95"
for g in groupby(df, :estimand)
    t = g.truth[1]
    cover = mean(abs.(g.estimate .- t) .<= 1.96 .* g.se)
    @printf "%-10s %9.4f %9.4f %9.4f %9.4f %9.4f %9.3f\n" g.estimand[1] t mean(g.estimate) - t std(g.estimate) mean(g.se) sqrt(mean((g.estimate .- t) .^ 2)) cover
end
