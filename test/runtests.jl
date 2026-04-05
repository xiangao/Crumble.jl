using Crumble
using Test
using DataFrames

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
        vars = CrumbleVars(["A"], :Y, ["M"], ["Z"], ["W"])
        @test vars.A == [:A]
        @test vars.Y == :Y
        @test vars.M == [:M]
    end

    @testset "is_binary" begin
        @test is_binary([0, 1, 0, 1]) == true
        @test is_binary([0, 1, 2]) == false
        @test is_binary([0.0, 1.0, missing]) == true
    end

    @testset "normalize_weights" begin
        @test isapprox(normalize_weights([2.0, 2.0, 2.0]), [1.0, 1.0, 1.0])
        @test isapprox(normalize_weights([1.0, 1.0]), [1.0, 1.0])
    end

    @testset "make_folds" begin
        folds = make_folds(100, 5)
        @test length(folds) == 5
        all_train = reduce(vcat, [f.training_set for f in folds])
        all_valid = reduce(vcat, [f.validation_set for f in folds])
        @test sort(all_valid) == 1:100
    end

    @testset "calc_estimates_natural" begin
        eif_ns = Dict(
            "100" => Dict("estimate" => 0.5, "std.error" => 0.1),
            "000" => Dict("estimate" => 0.2, "std.error" => 0.1),
            "111" => Dict("estimate" => 0.8, "std.error" => 0.1),
        )
        results = calc_estimates_natural(eif_ns, ones(100))
        @test results["direct"]["estimate"] ≈ 0.3
        @test results["indirect"]["estimate"] ≈ 0.3
        @test results["ate"]["estimate"] ≈ 0.6
    end

    @testset "assert_effect_type" begin
        @test assert_effect_type(["Z"], "RT") == true
        @test assert_effect_type(["Z"], "RI") == true
        @test assert_effect_type(nothing, "N") == true
        @test_throws ArgumentError assert_effect_type(nothing, "RT")
        @test_throws ArgumentError assert_effect_type(["Z"], "N")
    end

    @testset "shift functions" begin
        data = DataFrame(A=[0.5, 0.6, 0.7], Y=[1, 0, 1])
        shifted = shift_data(data, [:A], nothing, (d, t) -> d[:, t] .+ 0.1)
        @test shifted[:, :A] ≈ [0.6, 0.7, 0.8]
    end
end
