struct CrumbleVars
    A::Vector{Symbol}
    Y::Symbol
    M::Vector{Symbol}
    Z::Vector{Symbol}
    W::Vector{Symbol}
    C::Union{Symbol, Nothing}
    id::Union{Symbol, Nothing}

    function CrumbleVars(A, Y, M, Z, W, C=nothing, id=nothing)
        A = Symbol.(A)
        M = Symbol.(M)
        Z = Symbol.(Z)
        W = Symbol.(W)
        C = C === nothing ? nothing : Symbol(C)
        id = id === nothing ? nothing : Symbol(id)
        Y isa Symbol && (Y = [Y])
        Y isa String && (Y = [Symbol(Y)])
        Y = Symbol.(Y)
        length(Y) == 1 || throw(ArgumentError("Y must be a single symbol"))
        Y = Y[1]
        new(A, Y, M, Z, W, C, id)
    end
end

mutable struct CrumbleData
    data::DataFrame
    vars::CrumbleVars
    weights::Vector{Float64}
    d0::Union{Function, Nothing}
    d1::Union{Function, Nothing}
    data_0::DataFrame
    data_1::DataFrame
    data_0zp::DataFrame
    data_1zp::DataFrame

    function CrumbleData(data::DataFrame, vars::CrumbleVars, weights, d0=nothing, d1=nothing)
        if !isempty(vars.Z)
            z_ohe = one_hot_encode(data, vars.Z)
            vars = CrumbleVars(vars.A, vars.Y, vars.M, Symbol.(names(z_ohe)), vars.W, vars.C, vars.id)
            data = select(data, Not(vars.Z))
            data = hcat(data, z_ohe)
        end

        weights = normalize_weights(weights)
        data_0 = shift_data(data, vars.A, vars.C, d0)
        data_1 = shift_data(data, vars.A, vars.C, d1)
        data_0zp = DataFrame()
        data_1zp = DataFrame()

        new(data, vars, weights, d0, d1, data_0, data_1, data_0zp, data_1zp)
    end
end

struct CrumbleControl
    crossfit_folds::Int
    mlr3superlearner_folds::Int
    zprime_folds::Int
    epochs::Int
    learning_rate::Float64
    batch_size::Int
    device::String
end

struct CrumbleResult
    estimates::Dict{String, Dict{String, Any}}
    outcome_reg::Dict{String, Any}
    alpha_n::Union{Dict{String, Any}, Nothing}
    alpha_r::Union{Dict{String, Any}, Nothing}
    fits::Dict{String, Any}
    effect::String
end

struct CrossFitFold
    training_set::Vector{Int}
    validation_set::Vector{Int}
end

function training(cd::CrumbleData, fold_obj::Vector{CrossFitFold}, fold::Int)
    idx = fold_obj[fold].training_set
    (
        data = cd.data[idx, :],
        data_0 = cd.data_0[idx, :],
        data_1 = cd.data_1[idx, :],
        data_0zp = isempty(cd.data_0zp) ? DataFrame() : cd.data_0zp[idx, :],
        data_1zp = isempty(cd.data_1zp) ? DataFrame() : cd.data_1zp[idx, :],
    )
end

function validation(cd::CrumbleData, fold_obj::Vector{CrossFitFold}, fold::Int)
    idx = fold_obj[fold].validation_set
    (
        data = cd.data[idx, :],
        data_0 = cd.data_0[idx, :],
        data_1 = cd.data_1[idx, :],
        data_0zp = isempty(cd.data_0zp) ? DataFrame() : cd.data_0zp[idx, :],
        data_1zp = isempty(cd.data_1zp) ? DataFrame() : cd.data_1zp[idx, :],
    )
end
