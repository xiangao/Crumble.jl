struct CrumbleVars
    A::Vector{Symbol}
    Y::Symbol
    M::Vector{Symbol}
    Z::Vector{Symbol}
    W::Vector{Symbol}
    C::Union{Symbol, Nothing}
    id::Union{Symbol, Nothing}

    function CrumbleVars(A, Y, M, Z, W, C = nothing, id = nothing)
        Y isa AbstractVector && (length(Y) == 1 || throw(ArgumentError("Y must be a single column")); Y = Y[1])
        new(Symbol.(A), Symbol(Y), Symbol.(M), Symbol.(Z), Symbol.(W),
            C === nothing ? nothing : Symbol(C), id === nothing ? nothing : Symbol(id))
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

    function CrumbleData(data::DataFrame, vars::CrumbleVars, weights, d0 = nothing, d1 = nothing)
        # As in R crumble, only non-numeric mediator-outcome confounders are one-hot
        # encoded; numeric ones are used as they are.
        factor_z = [z for z in vars.Z if !(eltype(data[!, z]) <: Union{Missing, Real})]
        if !isempty(factor_z)
            z_ohe = one_hot_encode(data, factor_z)
            data = hcat(DataFrames.select(data, Not(factor_z)), z_ohe)
            vars = CrumbleVars(vars.A, vars.Y, vars.M,
                               [[z for z in vars.Z if !(z in factor_z)]; Symbol.(names(z_ohe))],
                               vars.W, vars.C, vars.id)
        end
        weights = normalize_weights(Float64.(weights))
        data_0 = shift_data(data, vars.A, vars.C, d0)
        data_1 = shift_data(data, vars.A, vars.C, d1)
        new(data, vars, weights, d0, d1, data_0, data_1, DataFrame(), DataFrame())
    end
end

struct CrumbleControl
    crossfit_folds::Int
    mlr3superlearner_folds::Int   # kept for API parity with R; no super learner is fitted
    zprime_folds::Int
    epochs::Int
    learning_rate::Float64
    batch_size::Int
    device::String
    riesz::Symbol                 # :nn (R crumble's default) or :linear
    riesz_basis::Symbol           # for :linear — :main or :saturated
    riesz_ridge::Float64          # for :linear
    alpha_cap::Float64            # upper bound on every fitted representer (Inf = none)
end

struct CrumbleResult
    estimates::Dict{String, Dict{String, Any}}
    outcome_reg::Dict{String, Any}
    alpha_n::Union{Dict{String, Dict{String, Vector{Float64}}}, Nothing}
    alpha_r::Union{Dict{String, Dict{String, Vector{Float64}}}, Nothing}
    effect::String
end

struct CrossFitFold
    training_set::Vector{Int}
    validation_set::Vector{Int}
end

function subset_data(cd::CrumbleData, idx)
    (data = cd.data[idx, :],
     data_0 = cd.data_0[idx, :],
     data_1 = cd.data_1[idx, :],
     data_0zp = isempty(cd.data_0zp) ? DataFrame() : cd.data_0zp[idx, :],
     data_1zp = isempty(cd.data_1zp) ? DataFrame() : cd.data_1zp[idx, :])
end

training(cd::CrumbleData, folds::Vector{CrossFitFold}, v::Int) = subset_data(cd, folds[v].training_set)
validation(cd::CrumbleData, folds::Vector{CrossFitFold}, v::Int) = subset_data(cd, folds[v].validation_set)
