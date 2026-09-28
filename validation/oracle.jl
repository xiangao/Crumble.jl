# Oracle check of the influence functions. Every nuisance — the sequential outcome
# regressions and the sequential Riesz representers — is replaced by its true value
# under the DGP of dgp.jl, so the only randomness is the sample and the Z' draw.
# If the influence functions are right, the oracle estimator is unbiased and its
# IF standard error matches its sampling sd.
# Run: julia -t 6 --project=validation validation/oracle.jl <n> <reps>
using Crumble, DataFrames, Random, Statistics, Printf, LinearAlgebra
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "dgp.jl"))

n, reps = parse(Int, ARGS[1]), parse(Int, ARGS[2])

pw(w1, w2, w3) = bern(pw1, w1) * bern(pw2, w2) * bern(pw3(w1, w2), w3)
paw(a, w...) = pw(w...) * bern(pa(w...), a)
pazw(a, z, w...) = paw(a, w...) * bern(pz(a, w...), z)
pazmw(a, z, m, w...) = pazw(a, z, w...) * bern(pm(a, z, w[1], w[2]), m)
pamw(a, m, w...) = sum(pazmw(a, z, m, w...) for z in 0:1)
pm_azw(m, a, z, w...) = bern(pm(a, z, w[1], w[2]), m)
pz_aw(z, a, w...) = bern(pz(a, w...), z)
mu(a, z, m, w...) = py(a, z, m, w...)

# natural functional (j, k, l)
function eif_nat(d, j, k, l)
    th2(a, z, w...) = sum(pm_azw(m, a, z, w...) * mu(j, z, m, w...) for m in 0:1)
    th1(a, w...) = sum(pz_aw(z, a, w...) * th2(k, z, w...) for z in 0:1)
    a1(a, w...) = a == l ? pw(w...) / paw(l, w...) : 0.0
    a2(a, z, w...) = a == k ? sum(a1(b, w...) * pazw(b, z, w...) for b in 0:1) / pazw(k, z, w...) : 0.0
    a3(a, z, m, w...) = a == j ? sum(a2(b, z, w...) * pazmw(b, z, m, w...) for b in 0:1) / pazmw(j, z, m, w...) : 0.0
    [begin
        w = (r.W_1, r.W_2, r.W_3); A, Z, M, Y = r.A, r.Z, r.M, r.Y
        a3(A, Z, M, w...) * (Y - mu(A, Z, M, w...)) +
        a2(A, Z, w...) * (mu(j, Z, M, w...) - th2(A, Z, w...)) +
        a1(A, w...) * (th2(k, Z, w...) - th1(A, w...)) + th1(l, w...)
     end for r in eachrow(d)]
end

# randomized functional (i, j, k, l); zp = the Z' column
function eif_ran(d, zp, i, j, k, l)
    th3(a, m, w...) = sum(pz_aw(z, a, w...) * mu(i, z, m, w...) for z in 0:1)
    th2(a, z, w...) = sum(pm_azw(m, a, z, w...) * th3(j, m, w...) for m in 0:1)
    th1(a, w...) = sum(pz_aw(z, a, w...) * th2(k, z, w...) for z in 0:1)
    a1(a, w...) = a == l ? pw(w...) / paw(l, w...) : 0.0
    a2(a, z, w...) = a == k ? sum(a1(b, w...) * pazw(b, z, w...) for b in 0:1) / pazw(k, z, w...) : 0.0
    a3(a, m, w...) = a == j ? sum(a2(b, z, w...) * pazmw(b, z, m, w...) for b in 0:1, z in 0:1) / pamw(j, m, w...) : 0.0
    a4(a, z, m, w...) = a == i ? sum(a3(b, m, w...) * pamw(b, m, w...) * pz_aw(z, b, w...) for b in 0:1) / pazmw(i, z, m, w...) : 0.0
    [begin
        r = d[t, :]; w = (r.W_1, r.W_2, r.W_3); A, Z, M, Y = r.A, r.Z, r.M, r.Y
        a4(A, Z, M, w...) * (Y - mu(A, Z, M, w...)) +
        a3(A, M, w...) * (mu(i, zp[t], M, w...) - th3(A, M, w...)) +
        a2(A, Z, w...) * (th3(j, M, w...) - th2(A, Z, w...)) +
        a1(A, w...) * (th2(k, Z, w...) - th1(A, w...)) + th1(l, w...)
     end for t in 1:nrow(d)]
end

T = truth()
contrasts = Dict(
    "RI_ride" => (["r1100"], ["r0000"]), "RI_riie" => (["r1111"], ["r1100"]),
    "RT_ate" => (["n111"], ["n000"]), "RT_p1" => (["n111"], ["n011"]),
    "RT_p2" => (["r0111"], ["r0011"]), "RT_p3" => (["r0011"], ["r0010"]),
    "RT_p4" => (["n010"], ["n000"]))
keys_ = sort(collect(keys(contrasts)))
est = zeros(reps, length(keys_)); se = zeros(reps, length(keys_))
Threads.@threads for rep in 1:reps
    rng = Xoshiro(20_000 + rep)
    d = make_example_data(n; rng = rng)
    AW = Float64.(Matrix(d[:, [:A, :W_1, :W_2, :W_3]]))
    Random.seed!(30_000 + rep)
    perm = Crumble.derangement_by_distance(AW)
    zp = d.Z[perm]
    f = Dict{String, Vector{Float64}}()
    for s in ("111", "011", "010", "000"); f["n"*s] = eif_nat(d, parse.(Int, collect(s))...); end
    for s in ("1100", "0000", "1111", "0111", "0011", "0010"); f["r"*s] = eif_ran(d, zp, parse.(Int, collect(s))...); end
    for (c, key) in enumerate(keys_)
        (p, q) = contrasts[key]
        ifv = f[p[1]] .- f[q[1]]
        est[rep, c] = mean(ifv); se[rep, c] = std(ifv) / sqrt(n)
    end
end
@printf "oracle n=%d reps=%d\n%-8s %9s %9s %9s %9s %8s\n" n reps "estimand" "truth" "bias" "sd" "mean_se" "cover"
for (c, key) in enumerate(keys_)
    t = T[key]
    @printf "%-8s %9.4f %9.4f %9.4f %9.4f %8.3f\n" key t mean(est[:, c]) - t std(est[:, c]) mean(se[:, c]) mean(abs.(est[:, c] .- t) .<= 1.96 .* se[:, c])
end
