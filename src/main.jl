function sequential_module(; layers::Int = 1, hidden::Int = 20, dropout::Float64 = 0.1)
    return function (d_in::Int)
        Chain(
            Dense(d_in, hidden, elu),
            (layers > 0 ? Chain([Dense(hidden, hidden, elu) for _ in 1:layers]...) : identity),
            Dense(hidden, 1),
            Dropout(dropout),
            softplus,
        )
    end
end

"""
    crumble_control(; crossfit_folds=10, zprime_folds=1, epochs=100, learning_rate=0.01,
                    batch_size=64, riesz=:nn, riesz_basis=:main, riesz_ridge=1e-8)

`riesz = :nn` fits the Riesz representers with R crumble's neural network.
`riesz = :linear` fits them in closed form over a basis; `riesz_basis = :saturated`
uses one indicator per covariate pattern, the exact empirical representer when every
variable is discrete.

`alpha_cap` bounds every fitted representer from above, the analogue of truncating
a propensity score. Under weak overlap a training fold can miss a covariate pattern
that the shift needs; the Riesz loss then has no minimiser there, and a flexible fit
drifts to arbitrarily large values. Choose a cap well above the plausible size of the
true weights; the default (`Inf`) applies none.
"""
function crumble_control(;
    crossfit_folds::Int = 10,
    mlr3superlearner_folds::Int = 10,
    zprime_folds::Int = 1,
    epochs::Int = 100,
    learning_rate::Float64 = 0.01,
    batch_size::Int = 64,
    device::String = "cpu",
    riesz::Symbol = :nn,
    riesz_basis::Symbol = :main,
    riesz_ridge::Float64 = 1e-8,
    alpha_cap::Float64 = Inf,
)
    riesz in (:nn, :linear) || throw(ArgumentError("riesz must be :nn or :linear"))
    riesz_basis in (:main, :saturated) || throw(ArgumentError("riesz_basis must be :main or :saturated"))
    CrumbleControl(crossfit_folds, mlr3superlearner_folds, zprime_folds, epochs, learning_rate,
                   batch_size, device, riesz, riesz_basis, riesz_ridge, alpha_cap)
end

"""
    crumble(data, trt; outcome, mediators, covar, moc=nothing, obs=nothing, id=nothing,
            d0=nothing, d1=nothing, effect="RT", weights=ones(nrow(data)),
            learners=["glm"], nn_module=sequential_module(), control=crumble_control())

Natural (`"N"`), organic (`"O"`), randomized interventional (`"RI"`) and
recanting-twin (`"RT"`) effects by cross-fitted one-step estimation with Riesz
representers, following the R package `crumble`.

`d0`/`d1` are shift functions `(data, trt) -> new treatment values`. When omitted
and the treatment is a single 0/1 variable, they default to setting it to 0 and 1.
(In R crumble an omitted shift leaves the treatment unchanged, which makes every
effect zero.)
"""
function crumble(
    data::DataFrame,
    trt::Vector{String};
    outcome::String,
    mediators::Vector{String},
    moc::Union{Vector{String}, Nothing} = nothing,
    covar::Vector{String},
    obs::Union{String, Nothing} = nothing,
    id::Union{String, Nothing} = nothing,
    d0::Union{Function, Nothing} = nothing,
    d1::Union{Function, Nothing} = nothing,
    effect::String = "RT",
    weights::AbstractVector{<:Real} = ones(nrow(data)),
    learners::Vector{String} = ["glm"],
    nn_module = sequential_module(),
    control::CrumbleControl = crumble_control(),
)
    effect = uppercase(effect)
    effect in ("RT", "N", "RI", "O") || throw(ArgumentError("effect must be one of RT, N, RI, O"))
    length(learners) == 1 || throw(ArgumentError(
        "Crumble.jl fits a single learner; a super learner over several libraries is not implemented."))
    length(weights) == nrow(data) || throw(ArgumentError("weights must have one entry per row"))

    assert_not_missing(data, trt, covar, mediators, moc === nothing ? String[] : moc,
                       obs === nothing ? String[] : [obs])
    assert_binary_0_1(data, Symbol(outcome))
    assert_binary_0_1(data, obs === nothing ? nothing : Symbol(obs))
    assert_effect_type(moc, effect)

    if d0 === nothing || d1 === nothing
        (length(trt) == 1 && is_binary(data[:, trt[1]])) || throw(ArgumentError(
            "Crumble.jl: supply d0 and d1 unless the treatment is a single 0/1 variable."))
        d0 = something(d0, (d, a) -> zeros(nrow(d)))
        d1 = something(d1, (d, a) -> ones(nrow(d)))
    end

    params = Dict("N" => natural, "O" => organic, "RT" => recanting_twin, "RI" => randomized)[effect]
    vars = CrumbleVars(trt, outcome, mediators, moc === nothing ? String[] : moc, covar, obs, id)
    cd = add_zp(CrumbleData(data, vars, weights, d0, d1), moc, control)

    folds = make_folds(nrow(cd.data), control.crossfit_folds,
                       id === nothing ? nothing : cd.data[:, cd.vars.id],
                       is_binary(skipmissing(cd.data[:, cd.vars.Y])) ? cd.data[:, cd.vars.Y] : nothing)

    thetas = estimate_theta(cd, folds, params, learners[1])
    alpha_ns = estimate_phi_n_alpha(cd, folds, params, nn_module, control)
    alpha_rs = estimate_phi_r_alpha(cd, folds, params, nn_module, control)
    eif_ns = calc_eifs(cd, alpha_ns, thetas, "n", eif_n)
    eif_rs = calc_eifs(cd, alpha_rs, thetas, "r", eif_r)

    estimates = effect == "N"  ? calc_estimates_natural(eif_ns) :
                effect == "O"  ? calc_estimates_organic(eif_ns) :
                effect == "RI" ? calc_estimates_ri(eif_rs) :
                                 calc_estimates_rt(eif_ns, eif_rs)

    CrumbleResult(estimates, Dict{String, Any}(thetas), alpha_ns, alpha_rs, effect)
end
