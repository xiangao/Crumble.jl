using Flux
using Optimisers

function estimate_phi_n_alpha(cd::CrumbleData, folds::Vector{CrossFitFold}, params::Dict, nn_module, control::CrumbleControl)
    if !haskey(params, :natural) || isempty(params[:natural])
        return nothing
    end

    alpha_ns = Vector{Any}(undef, control.crossfit_folds)
    @showprogress desc="Computing alpha n density ratios..." for i in 1:control.crossfit_folds
        train = training(cd, folds, i)
        valid = validation(cd, folds, i)

        alpha_ns[i] = Dict{String, Any}()
        for param in params[:natural]
            result = phi_n_alpha(train, valid, cd.vars, nn_module, param, control)
            key = replace("$(param["j"])$(param["k"])$(param["l"])", "data_" => "")
            alpha_ns[i][key] = result
        end
    end

    return recombine_alpha(alpha_ns, folds)
end

function phi_n_alpha(train, valid, vars::CrumbleVars, architecture, params, control::CrumbleControl)
    j = params["j"]
    k = params["k"]
    l = params["l"]

    n_train = nrow(train.data)
    n_valid = nrow(valid.data)

    vars1 = [vars.A..., vars.W...]
    vars1 = [v for v in vars1 if v !== nothing && hasproperty(train.data, v)]
    
    if isempty(vars1)
        return Dict("jkl" => "", "alpha1" => ones(n_valid), "alpha2" => ones(n_valid), "alpha3" => ones(n_valid))
    end
    
    X_train = Matrix{Float64}(train.data[:, vars1])
    X_valid = Matrix{Float64}(valid.data[:, vars1])
    
    model = architecture(size(X_train, 2))
    opt_state = Flux.setup(Adam(control.learning_rate), model)
    
    function loss_fn(m, x)
        preds = m(x')
        return mean(preds.^2)
    end
    
    for epoch in 1:control.epochs
        batch_idx = rand(1:n_train, min(control.batch_size, n_train))
        X_batch = X_train[batch_idx, :]
        
        grads = Flux.gradient(model) do m
            loss_fn(m, X_batch)
        end
        Flux.update!(opt_state, model, grads[1])
    end
    
    Flux.testmode!(model)
    
    alpha1_valid = vec(model(X_valid')[:, 1])
    # QUARANTINED: these two components were previously filled with random
    # perturbations of one. Random numbers are not estimated Riesz representers,
    # and anything computed from them is not an estimate of the target
    # functional. Refuse to continue rather than return a number that looks real.
    error("""
          Crumble.jl: the Riesz representers alpha2 and alpha3 are not implemented.
          The previous code substituted random values here, which silently produced
          invalid estimates. See the prototype warning in the README.
          """)
    
    jkl = replace("$(j)$(k)$(l)", "data_" => "")
    return Dict(
        "jkl" => jkl,
        "alpha1" => alpha1_valid,
        "alpha2" => alpha2_valid,
        "alpha3" => alpha3_valid,
    )
end

function estimate_phi_r_alpha(cd::CrumbleData, folds::Vector{CrossFitFold}, params::Dict, nn_module, control::CrumbleControl)
    if !haskey(params, :randomized) || isempty(params[:randomized])
        return nothing
    end

    alpha_rs = Vector{Any}(undef, control.crossfit_folds)
    @showprogress desc="Computing alpha r density ratios..." for i in 1:control.crossfit_folds
        train = training(cd, folds, i)
        valid = validation(cd, folds, i)

        alpha_rs[i] = Dict{String, Any}()
        for param in params[:randomized]
            # QUARANTINED: all four components were independent uniform random
            # numbers. See the note above; this route is not implemented.
            error("""
                  Crumble.jl: the randomized-interventional Riesz representers are not
                  implemented. The previous code substituted uniform random values.
                  See the prototype warning in the README.
                  """)
            key = replace("$(param["i"])$(param["j"])$(param["k"])$(param["l"])", "data_" => "", "zp" => "")
            alpha_rs[i][key] = result
        end
    end

    return recombine_alpha(alpha_rs, folds)
end
