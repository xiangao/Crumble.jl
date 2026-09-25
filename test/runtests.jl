# Run Crumble unit tests
using Crumble
using Test
using DataFrames
using Statistics
using Flux

@testset "Crumble.jl" begin
    @testset "crumble_control" begin
        ctrl = crumble_control(crossfit_folds=5, epochs=10)
        @test ctrl.crossfit_folds == 5
        @test ctrl.epochs == 10
        @test ctrl.learning_rate == 0.01
        @test ctrl.batch_size == 64
    end

    @testset "sequential_module" begin
        module_fn = sequential_module(layers=2, hidden=32, dropout=0.2)
        model = module_fn(10)
        @test model isa Flux.Chain
    end

    @testset "CrumbleVars" begin
        vars = Crumble.CrumbleVars(["A"], :Y, ["M"], ["Z"], ["W"])
        @test vars.A == [:A]
        @test vars.Y == :Y
        @test vars.M == [:M]
    end

    @testset "is_binary" begin
        @test Crumble.is_binary([0, 1, 0, 1]) == true
        @test Crumble.is_binary([0, 1, 2]) == false
        @test Crumble.is_binary([0.0, 1.0, missing]) == true
    end

    @testset "normalize_weights" begin
        @test isapprox(Crumble.normalize_weights([2.0, 2.0, 2.0]), [1.0, 1.0, 1.0])
        @test isapprox(Crumble.normalize_weights([1.0, 1.0]), [1.0, 1.0])
    end

    @testset "make_folds" begin
        folds = Crumble.make_folds(100, 5)
        @test length(folds) == 5
        all_train = reduce(vcat, [f.training_set for f in folds])
        all_valid = reduce(vcat, [f.validation_set for f in folds])
        @test sort(all_valid) == 1:100
    end

    @testset "calc_estimates_natural" begin
        # Influence functions are required. Build three whose means are 0.5, 0.2
        # and 0.8 and which are correlated across observations, so that the
        # covariance term in the contrast actually matters.
        n = 100
        base = collect(range(-1.0, 1.0, length = n))
        base = base .- mean(base)
        if_100 = 0.5 .+ base
        if_000 = 0.2 .+ base          # perfectly correlated with if_100
        if_111 = 0.8 .+ 2.0 .* base
        eif_ns = Dict{String, Any}(
            "100" => Dict("estimate" => mean(if_100), "std.error" => std(if_100)/sqrt(n), "influence" => if_100),
            "000" => Dict("estimate" => mean(if_000), "std.error" => std(if_000)/sqrt(n), "influence" => if_000),
            "111" => Dict("estimate" => mean(if_111), "std.error" => std(if_111)/sqrt(n), "influence" => if_111),
        )
        w = ones(n)
        results = Crumble.calc_estimates_natural(eif_ns, w)
        @test results["direct"]["estimate"] ≈ 0.3
        @test results["indirect"]["estimate"] ≈ 0.3
        @test results["ate"]["estimate"] ≈ 0.6

        # if_100 and if_000 differ by a constant, so their contrast influence
        # function is constant and the contrast is estimated without error. The
        # old sqrt(se_a^2 + se_b^2) formula would have reported a positive SE here.
        @test isnan(results["direct"]["std.error"])

        # For a contrast whose influence function is not degenerate, the SE must
        # equal sd(IF_a - IF_b)/sqrt(n), not the independence formula.
        se_correct = std(if_111 .- if_000) / sqrt(n)
        se_independence = sqrt((std(if_111)/sqrt(n))^2 + (std(if_000)/sqrt(n))^2)
        @test results["ate"]["std.error"] ≈ se_correct
        @test !isapprox(results["ate"]["std.error"], se_independence; rtol = 1e-6)
    end

    @testset "quarantined paths raise rather than fabricate" begin
        # Missing required functionals must not be silently substituted.
        @test_throws ErrorException Crumble.calc_estimates_natural(
            Dict{String, Any}("100" => Dict("estimate" => 0.1, "std.error" => 0.1,
                                            "influence" => zeros(10))), ones(10))
        # Empty input must not return a precise-looking zero.
        @test_throws ErrorException Crumble.calc_estimates_natural(Dict{String, Any}(), ones(10))
        # Unimplemented estimands must not return natural-effect numbers.
        @test_throws ErrorException Crumble.calc_estimates_organic(Dict{String, Any}(), ones(10))
        @test_throws ErrorException Crumble.calc_estimates_ri(Dict{String, Any}(), ones(10))
        @test_throws ErrorException Crumble.calc_estimates_rt(Dict{String, Any}(), Dict{String, Any}(), ones(10))
    end

    @testset "assert_effect_type" begin
        @test Crumble.assert_effect_type(["Z"], "RT") == true
        @test Crumble.assert_effect_type(["Z"], "RI") == true
        @test Crumble.assert_effect_type(nothing, "N") == true
        @test_throws ArgumentError Crumble.assert_effect_type(nothing, "RT")
        @test_throws ArgumentError Crumble.assert_effect_type(["Z"], "N")
    end

    @testset "shift functions" begin
        data = DataFrame(A=[0.5, 0.6, 0.7], Y=[1, 0, 1])
        shifted = Crumble.shift_data(data, [:A], nothing, (d, t) -> d[:, t] .+ 0.1)
        @test shifted[:, :A] ≈ [0.6, 0.7, 0.8]
    end
end
