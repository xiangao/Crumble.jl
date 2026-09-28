using Crumble
using Test
using DataFrames
using Statistics
using Random
using LinearAlgebra
using Flux

@testset "Crumble.jl" begin
    @testset "crumble_control" begin
        ctrl = crumble_control(crossfit_folds = 5, epochs = 10)
        @test ctrl.crossfit_folds == 5
        @test ctrl.epochs == 10
        @test ctrl.riesz == :nn
        @test_throws ArgumentError crumble_control(riesz = :forest)
    end

    @testset "sequential_module" begin
        @test sequential_module(layers = 2, hidden = 32, dropout = 0.2)(10) isa Flux.Chain
    end

    @testset "param_key" begin
        @test Crumble.param_key(Dict("j" => "data_1", "k" => "data_0", "l" => "data_0")) == "100"
        @test Crumble.param_key(Dict("i" => "data_1zp", "j" => "data_1", "k" => "data_0", "l" => "data_0")) == "1100"
    end

    @testset "make_folds" begin
        folds = Crumble.make_folds(100, 5)
        @test sort(reduce(vcat, [f.validation_set for f in folds])) == 1:100
        @test all(isempty(intersect(f.training_set, f.validation_set)) for f in folds)
        # clusters are never split across folds
        id = repeat(1:20, inner = 5)
        for f in Crumble.make_folds(100, 4, id)
            @test isempty(intersect(Set(id[f.training_set]), Set(id[f.validation_set])))
        end
    end

    @testset "hungarian matches brute force" begin
        rng = MersenneTwister(3)
        perms(v) = length(v) <= 1 ? [v] : [[v[i]; p] for i in eachindex(v) for p in perms(deleteat!(copy(v), i))]
        for n in 2:6, _ in 1:5
            C = rand(rng, n, n)
            p = Crumble.hungarian(C)
            @test sort(p) == 1:n
            @test sum(C[i, p[i]] for i in 1:n) ≈ minimum(sum(C[i, q[i]] for i in 1:n) for q in perms(collect(1:n)))
        end
    end

    @testset "Z' is a derangement matched on (A, W)" begin
        rng = MersenneTwister(4)
        AW = Float64.(hcat(rand(rng, 0:1, 60), rand(rng, 0:1, 60)))
        p = Crumble.derangement_by_distance(AW)
        @test sort(p) == 1:60
        @test all(p[i] != i for i in 1:60)                     # no unit keeps its own Z
        @test all(AW[i, :] == AW[p[i], :] for i in 1:60)        # every cell has >= 2 units here
    end

    @testset "saturated Riesz representer is the empirical density ratio" begin
        # alpha solving E[alpha(A,W) h(A,W)] = E[h(1,W)] is 1{A=1}/P(A=1|W).
        rng = MersenneTwister(5)
        n = 4000
        W = rand(rng, 0:2, n)
        A = Int.(rand(rng, n) .< 0.2 .+ 0.25 .* W)
        X = Float64.(hcat(A, W))
        Xs = Float64.(hcat(ones(n), W))
        ctrl = crumble_control(riesz = :linear, riesz_basis = :saturated, riesz_ridge = 0.0)
        a = Crumble.fit_alpha(X, Xs, ones(n), X, ctrl, nothing)
        for w in 0:2
            pw = mean(A[W .== w])
            @test all(a.train[(A .== 1) .& (W .== w)] .≈ 1 / pw)
            @test all(abs.(a.train[(A .== 0) .& (W .== w)]) .< 1e-10)
        end
    end

    @testset "network representer is not shrunk" begin
        # A weighted stage: the representer solving E[a h(A,Z,W)] = E[w h(0,Z,W)]
        # has mean E[w] = 1 (take h = 1). R crumble's defaults (dropout 0.1, weight
        # decay 0.01) give 0.957 here; the defaults used now give 1.007.
        rng = MersenneTwister(21)
        n = 4000
        W = rand(rng, 0:2, n); A = Int.(rand(rng, n) .< 0.1 .+ 0.3 .* W)
        Z = Int.(rand(rng, n) .< 0.2 .+ 0.3 .* A .+ 0.1 .* W)
        w1 = [A[i] == 1 ? 1 / mean(A[W .== W[i]]) : 0.0 for i in 1:n]
        X = Float64.(hcat(A, Z, W)); Xs = Float64.(hcat(zeros(n), Z, W))
        Random.seed!(22)
        a = Crumble.fit_alpha(X, Xs, w1, X, crumble_control(epochs = 100), sequential_module())
        @test abs(mean(a.train) - mean(w1)) < 0.025
    end

    @testset "alpha_cap bounds the representers" begin
        # A shift into a pattern with a single training observation gives a large
        # empirical representer; the cap must bound it on both train and valid rows.
        X = Float64.(hcat([ones(99); 0.0], zeros(100)))
        Xs = Float64.(hcat(zeros(100), zeros(100)))
        ctrl = crumble_control(riesz = :linear, riesz_basis = :saturated, riesz_ridge = 0.0, alpha_cap = 7.0)
        a = Crumble.fit_alpha(X, Xs, ones(100), X, ctrl, nothing)
        @test maximum(a.train) == 7.0 && maximum(a.valid) == 7.0
        uncapped = Crumble.fit_alpha(X, Xs, ones(100), X, crumble_control(riesz = :linear, riesz_basis = :saturated, riesz_ridge = 0.0), nothing)
        @test maximum(uncapped.train) ≈ 100.0
    end

    @testset "IF standard errors" begin
        n = 100
        base = collect(range(-1.0, 1.0, length = n)); base .-= mean(base)
        w = ones(n); id = collect(1:n)
        e100 = Crumble.IFEstimate(0.5 .+ base, w, id)
        e000 = Crumble.IFEstimate(0.2 .+ base, w, id)
        e111 = Crumble.IFEstimate(0.8 .+ 2 .* base, w, id)
        d = Crumble.summarize(e100 - e000)
        @test d["estimate"] ≈ 0.3
        @test isnan(d["std.error"])                              # constant contrast IF
        ate = Crumble.summarize(e111 - e000)
        @test ate["std.error"] ≈ std(e111.eif .- e000.eif) / sqrt(n)
        @test !isapprox(ate["std.error"], sqrt(var(e111.eif) / n + var(e000.eif) / n); rtol = 1e-6)
        # cluster-robust: duplicating every row inside its own cluster leaves the SE
        # of the cluster totals unchanged, while the naive SE would shrink by sqrt(2)
        e = Crumble.IFEstimate(0.8 .+ 2 .* base, w, id)
        e2 = Crumble.IFEstimate(repeat(0.8 .+ 2 .* base, inner = 2), ones(2n), repeat(id, inner = 2))
        @test Crumble.std_error(e2) ≈ Crumble.std_error(e) rtol = 1e-12
    end

    @testset "estimate combinations" begin
        n = 50; w = ones(n); id = collect(1:n)
        mk(x) = Crumble.IFEstimate(fill(x, n) .+ 0.01 .* randn(MersenneTwister(round(Int, 1000x)), n), w, id)
        nat = Dict(k => mk(v) for (k, v) in ("111" => 0.9, "011" => 0.6, "010" => 0.5, "000" => 0.2))
        ran = Dict(k => mk(v) for (k, v) in ("0111" => 0.55, "0011" => 0.45, "0010" => 0.4))
        rt = Crumble.calc_estimates_rt(nat, ran)
        parts = sum(rt[p]["estimate"] for p in ("p1", "p2", "p3", "p4", "intermediate_confounding"))
        @test parts ≈ rt["ate"]["estimate"]
        @test rt["direct"]["estimate"] + rt["indirect"]["estimate"] + rt["intermediate_confounding"]["estimate"] ≈ rt["ate"]["estimate"]
        @test_throws ErrorException Crumble.calc_estimates_ri(Dict("1100" => mk(0.1)))
        @test_throws ErrorException Crumble.calc_estimates_natural(nothing)
    end

    @testset "assert_effect_type" begin
        @test Crumble.assert_effect_type(["Z"], "RT")
        @test Crumble.assert_effect_type(nothing, "N")
        @test_throws ArgumentError Crumble.assert_effect_type(nothing, "RT")
        @test_throws ArgumentError Crumble.assert_effect_type(["Z"], "N")
    end

    @testset "shift functions" begin
        data = DataFrame(A = [0.5, 0.6, 0.7], Y = [1, 0, 1])
        @test Crumble.shift_data(data, [:A], nothing, (d, t) -> d[:, t] .+ 0.1)[:, :A] ≈ [0.6, 0.7, 0.8]
    end

    @testset "end to end: natural effects recover a known truth" begin
        # Binary W, A, M, Y with a saturated outcome model: the one-step estimator
        # is then consistent, and at n = 20000 must land within 4 SE of the truth
        # computed by enumeration.
        rng = MersenneTwister(11)
        lg(x) = 1 / (1 + exp(-x))
        pw = 0.5; pa(w) = lg(-0.5 + w); pm(a, w) = lg(-1 + 1.5a + 0.5w); py(a, m, w) = lg(-1 + 0.7a + 1.2m + 0.4w)
        n = 20000
        W = Int.(rand(rng, n) .< pw); A = Int.(rand(rng, n) .< pa.(W))
        M = Int.(rand(rng, n) .< pm.(A, W)); Y = Int.(rand(rng, n) .< py.(A, M, W))
        th(j, k) = sum((w == 1 ? pw : 1 - pw) * sum((m == 1 ? pm(k, w) : 1 - pm(k, w)) * py(j, m, w) for m in 0:1) for w in 0:1)
        truth = Dict("direct" => th(1, 0) - th(0, 0), "indirect" => th(1, 1) - th(1, 0), "ate" => th(1, 1) - th(0, 0))
        Random.seed!(12)
        r = crumble(DataFrame(W = W, A = A, M = M, Y = Y), ["A"]; outcome = "Y", mediators = ["M"],
                    covar = ["W"], effect = "N", learners = ["saturated"],
                    control = crumble_control(crossfit_folds = 5, riesz = :linear, riesz_basis = :saturated))
        for k in ("direct", "indirect", "ate")
            e = r.estimates[k]
            @test abs(e["estimate"] - truth[k]) < 4 * e["std.error"]
            @test 0 < e["std.error"] < 0.05
        end
    end
end
