using MLJBase
using MLJLinearModels
using ProgressMeter

function estimate_theta(cd::CrumbleData, folds::Vector{CrossFitFold}, params::Dict, learners::Vector{String}, control::CrumbleControl)
    thetas = Vector{Any}(undef, control.crossfit_folds)
    @showprogress desc="Fitting outcome regressions..." for i in 1:control.crossfit_folds
        train = training(cd, folds, i)
        valid = validation(cd, folds, i)
        thetas[i] = theta(train, valid, cd.vars, params, control)
    end
    return recombine_theta(thetas, folds)
end

function theta(train, valid, vars::CrumbleVars, params::Dict, control::CrumbleControl)
    n_valid = nrow(valid.data)
    result = Dict{String, Any}()
    
    all_cols = [vars.A..., vars.W..., vars.M..., vars.Z...]
    all_cols = [c for c in all_cols if hasproperty(train.data, c)]
    
    # Simple linear regression using backslash
    function fit_outcome(df::DataFrame, target_col::Symbol)
        y = Vector{Float64}(df[:, target_col])
        X = Matrix{Float64}(df[:, all_cols])
        θ = X \ y
        return θ
    end
    
    function predict_outcome(θ::Vector{Float64}, df::DataFrame)
        X = Matrix{Float64}(df[:, all_cols])
        return X * θ
    end
    
    # Theta Y: E[Y | A, W, M, Z]
    theta_y_coefs = fit_outcome(train.data, vars.Y)
    b3_train = predict_outcome(theta_y_coefs, train.data_1)
    b3_valid = predict_outcome(theta_y_coefs, valid.data_1)
    
    # Theta 2: b3 | A, W, Z
    cols_2 = [vars.A..., vars.W..., vars.Z...]
    cols_2 = [c for c in cols_2 if hasproperty(train.data, c)]
    
    function fit_theta2(df::DataFrame, pseudo_y::Vector{Float64}, col_names)
        y = pseudo_y
        X = Matrix{Float64}(df[:, col_names])
        θ = X \ y
        return θ
    end
    
    function predict_theta2(θ::Vector{Float64}, df::DataFrame, col_names)
        X = Matrix{Float64}(df[:, col_names])
        return X * θ
    end
    
    theta2_coefs = fit_theta2(train.data, b3_train, cols_2)
    b2_train = predict_theta2(theta2_coefs, train.data_0, cols_2)
    b2_valid = predict_theta2(theta2_coefs, valid.data_0, cols_2)
    
    # Theta 1: b2 | A, W
    cols_1 = [vars.A..., vars.W...]
    cols_1 = [c for c in cols_1 if hasproperty(train.data, c)]
    
    theta1_coefs = fit_theta2(train.data, b2_train, cols_1)
    b1_valid = predict_theta2(theta1_coefs, valid.data, cols_1)
    
    # Natural effects
    if haskey(params, :natural) && !isempty(params[:natural])
        vals_n = Dict{String, Any}()
        for param in params[:natural]
            j = param["j"]
            k = param["k"]
            l = param["l"]
            key = replace("$(j)$(k)$(l)", "data_" => "")
            vals_n[key] = Dict(
                "fit3_weights" => [1.0],
                "fit3_natural" => b3_train,
                "b3" => b3_valid,
                "fit2_weights" => [1.0],
                "fit2_natural" => b2_train,
                "b2" => b2_valid,
                "fit1_weights" => [1.0],
                "fit1_natural" => b2_train,
                "b1" => b1_valid,
            )
        end
        result["n"] = vals_n
    end
    
    # Randomized effects  
    if haskey(params, :randomized) && !isempty(params[:randomized])
        vals_r = Dict{String, Any}()
        for param in params[:randomized]
            i = param["i"]
            j = param["j"]
            k = param["k"]
            l = param["l"]
            key = replace("$(i)$(j)$(k)$(l)", "data_" => "")
            vals_r[key] = Dict(
                "b4" => b3_valid,
                "b3" => b2_valid,
                "b2" => b1_valid,
                "b1" => b1_valid,
            )
        end
        result["r"] = vals_r
    end
    
    return result
end
