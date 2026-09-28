# Variability of Crumble RI across seeds on one fixed dataset (n = 5000). The
# influence-function SE treats Z', the folds and the network initialisation as
# fixed, so any spread here is variance the reported SE does not include.
using Crumble, CSV, DataFrames, Random, Statistics, Printf, LinearAlgebra
BLAS.set_num_threads(1)
d = CSV.read(joinpath(@__DIR__, "results", "xcheck_n5000.csv"), DataFrame)
S = 12
est = zeros(S, 2); se = zeros(S, 2)
Threads.@threads for s in 1:S
    Random.seed!(s)
    r = crumble(d, ["A"]; outcome = "Y", mediators = ["M"], moc = ["Z"], covar = ["W_1", "W_2", "W_3"],
                effect = "RI", control = crumble_control(crossfit_folds = 5, alpha_cap = 100.0))
    est[s, :] = [r.estimates["ride"]["estimate"], r.estimates["riie"]["estimate"]]
    se[s, :] = [r.estimates["ride"]["std.error"], r.estimates["riie"]["std.error"]]
end
for (j, k) in enumerate(("ride", "riie"))
    @printf "%s: across-seed sd %.4f, mean reported se %.4f\n" k std(est[:, j]) mean(se[:, j])
end
