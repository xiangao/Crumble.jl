using ProgressMeter
using Distances

function add_zp(cd::CrumbleData, moc::Union{Vector{String}, Nothing}, control::CrumbleControl)
    if moc !== nothing
        zp = set_zp(cd, control.zprime_folds)
        cd.data_0zp = copy(cd.data_0)
        cd.data_1zp = copy(cd.data_1)
        for z in cd.vars.Z
            if hasproperty(cd.data_0zp, z)
                cd.data_0zp[:, z] .= zp[:, z]
            end
            if hasproperty(cd.data_1zp, z)
                cd.data_1zp[:, z] .= zp[:, z]
            end
        end
    end
    return cd
end

function linear_permutation(data::Matrix{Float64})
    n = size(data, 1)
    D = pairwise(Euclidean(), data, dims=1)
    D = D ./ maximum(D)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x[1:n, 1:n], Bin)
    @constraint(model, [i=1:n], sum(x[i, :]) == 1)
    @constraint(model, [j=1:n], sum(x[:, j]) == 1)
    @objective(model, Min, sum(D[i, j] * x[i, j] for i in 1:n, j in 1:n))

    optimize!(model)
    return value.(x)
end

function set_zp(cd::CrumbleData, folds::Int)
    fold_obj = make_folds(nrow(cd.data), folds, cd.id !== nothing ? cd.data[:, cd.id] : nothing)

    AW_cols = [cd.vars.A..., cd.vars.W...]
    AW_cols = [c for c in AW_cols if hasproperty(cd.data, c)]
    AW = Matrix{Float64}(cd.data[:, AW_cols])

    Z_cols = cd.vars.Z
    Z = cd.data[:, Z_cols]

    permuted = [Dict{Symbol, Vector{Float64}}() for _ in 1:length(fold_obj)]

    @showprogress desc="Permuting Z-prime variables..." for (i, fold) in enumerate(fold_obj)
        idx = fold.validation_set
        if length(idx) <= 1
            for z in names(Z)
                permuted[i][Symbol(z)] = Z[idx, z]
            end
            continue
        end

        P = linear_permutation(AW[idx, :])
        for z in names(Z)
            permuted[i][Symbol(z)] = vec(P * Matrix{Float64}(Z[idx, z]))
        end
    end

    result = Dict{Symbol, Vector{Float64}}(z => Vector{Float64}(undef, nrow(cd.data)) for z in names(Z))
    for z in names(Z)
        result[Symbol(z)] .= missing
    end

    for (i, fold) in enumerate(fold_obj)
        idx = fold.validation_set
        for z in names(Z)
            result[Symbol(z)][idx] = permuted[i][Symbol(z)]
        end
    end

    return DataFrame(result)
end
