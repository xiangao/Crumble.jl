function sequential_module(; layers::Int=1, hidden::Int=20, dropout::Float64=0.1)
    return function(d_in::Int)
        Chain(
            Dense(d_in, hidden, elu),
            (layers > 0 ? Chain([Chain(Dense(hidden, hidden, elu)) for _ in 1:layers]...) : identity),
            Dense(hidden, 1),
            Dropout(dropout),
            softplus,
        )
    end
end

function crumble_control(;
    crossfit_folds::Int=10,
    mlr3superlearner_folds::Int=10,
    zprime_folds::Int=1,
    epochs::Int=100,
    learning_rate::Float64=0.01,
    batch_size::Int=64,
    device::String="cpu",
)
    return CrumbleControl(
        crossfit_folds,
        mlr3superlearner_folds,
        zprime_folds,
        epochs,
        learning_rate,
        batch_size,
        device,
    )
end

function crumble(
    data::DataFrame,
    trt::Vector{String};
    outcome::String,
    mediators::Vector{String},
    moc::Union{Vector{String}, Nothing}=nothing,
    covar::Vector{String},
    obs::Union{String, Nothing}=nothing,
    id::Union{String, Nothing}=nothing,
    d0::Union{Function, Nothing}=nothing,
    d1::Union{Function, Nothing}=nothing,
    effect::String="RT",
    weights::Vector{Float64}=ones(nrow(data)),
    learners::Vector{String}=["glm"],
    nn_module=sequential_module(),
    control::CrumbleControl=crumble_control(),
)
    effect = uppercase(effect)
    @assert effect in ("RT", "N", "RI", "O") "effect must be one of RT, N, RI, O"

    assert_not_missing(data, trt, covar, mediators, moc !== nothing ? moc : String[], obs !== nothing ? [obs] : String[])
    assert_binary_0_1(data, Symbol(outcome))
    assert_binary_0_1(data, obs !== nothing ? Symbol(obs) : nothing)
    assert_effect_type(moc, effect)

    params = Dict(
        "N" => natural,
        "O" => organic,
        "RT" => recanting_twin,
        "RI" => randomized,
    )[effect]

    vars = CrumbleVars(
        trt,
        outcome,
        mediators,
        moc !== nothing ? moc : String[],
        covar,
        obs,
        id,
    )

    cd = CrumbleData(data, vars, weights, d0, d1)
    cd = add_zp(cd, moc, control)

    strata = outcome !== nothing ? Vector(cd.data[:, Symbol(outcome)]) : nothing
    folds = make_folds(nrow(cd.data), control.crossfit_folds,
                       id !== nothing ? Vector(cd.data[:, Symbol(id)]) : nothing,
                       strata)

    thetas = estimate_theta(cd, folds, params, learners, control)

    alpha_ns = estimate_phi_n_alpha(cd, folds, params, nn_module, control)
    eif_ns = calc_eifs(cd, alpha_ns, thetas, eif_n)

    alpha_rs = estimate_phi_r_alpha(cd, folds, params, nn_module, control)
    eif_rs = calc_eifs(cd, alpha_rs, thetas, eif_r)

    estimates = if effect == "N"
        calc_estimates_natural(eif_ns, weights)
    elseif effect == "O"
        calc_estimates_organic(eif_ns, weights)
    elseif effect == "RT"
        calc_estimates_rt(eif_ns, eif_rs, weights)
    elseif effect == "RI"
        calc_estimates_ri(eif_rs, weights)
    end

    return CrumbleResult(
        estimates,
        Dict("theta_n" => Dict(), "theta_r" => Dict()),
        alpha_ns,
        alpha_rs,
        Dict("theta_n" => Dict(), "theta_r" => Dict()),
        effect,
    )
end
