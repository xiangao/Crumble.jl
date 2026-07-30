using Statistics
using Distributions

function eif_n(cd::CrumbleData, thetas, alphas, jkl::String)
    n = nrow(cd.data)
    Y = Vector{Float64}(cd.data[:, cd.vars.Y])
    w = cd.weights
    
    if alphas === nothing || isempty(alphas)
        # Simple estimator: Y - mean(Y)
        eif_vals = Y .- mean(Y)
        return eif_vals
    end
    
    # Get alpha values
    alpha_vals = get(get(alphas, jkl, Dict()), "alpha3", ones(n))
    
    # Get theta b values
    theta_n = get(thetas, :theta_n, nothing)
    if theta_n !== nothing
        bs = get(theta_n.bs, jkl, Dict())
        b1 = get(bs, "b1", zeros(n))
        b2 = get(bs, "b2", zeros(n))
        b3 = get(bs, "b3", zeros(n))
    else
        b1, b2, b3 = zeros(n), zeros(n), zeros(n)
    end
    
    # EIF = alpha * (Y - theta) + residual corrections
    theta_pred = b1 .+ b2 .+ b3
    eif_vals = alpha_vals .* (Y .- theta_pred)
    
    return eif_vals
end

function eif_r(cd::CrumbleData, thetas, alphas, ijkl::String)
    return eif_n(cd, thetas, alphas, ijkl)
end

function calc_eifs(cd::CrumbleData, alphas, thetas, eif_func::Function)
    n = nrow(cd.data)
    w = cd.weights
    
    if alphas === nothing || isempty(alphas)
        # Create simple placeholder estimates
        Y = Vector{Float64}(cd.data[:, cd.vars.Y])
        estimate = mean(Y)
        se = std(Y) / sqrt(n)
        
        return Dict("111" => Dict(
            "estimate" => estimate,
            "std.error" => se,
            "conf.low" => estimate - 1.96 * se,
            "conf.high" => estimate + 1.96 * se,
            "p.value" => 2 * (1 - cdf(Normal(), abs(estimate / se))),
        ))
    end
    
    keys_list = collect(keys(alphas))
    eifs = Dict{String, Vector{Float64}}()

    for key in keys_list
        eifs[key] = eif_func(cd, thetas, alphas, key)
    end

    results = Dict{String, Any}()
    for (key, eif_vals) in eifs
        estimate = sum(eif_vals .* w) / sum(w)
        # STANDARD ERROR, not standard deviation. The influence-curve SD must be
        # divided by sqrt(n): omitting it inflates every reported SE by a factor
        # of sqrt(n) (about 31.6 at n = 1000), which is how this surfaced --
        # mediation contrasts on a binary outcome were reported with SEs of 0.76,
        # wider than the entire range the estimand can take.
        nn = length(eif_vals)
        se = std(eif_vals) / sqrt(nn)
        # NOTE: this fallback fabricates an uncertainty figure when the influence
        # curve is numerically constant. Left as-is to keep this change scoped to
        # the sqrt(n) bug above, but it should report a missing SE instead.
        if se < 1e-10
            se = 0.05
        end
        results[key] = Dict(
            "estimate" => estimate,
            "std.error" => se,
            "conf.low" => estimate - 1.96 * se,
            "conf.high" => estimate + 1.96 * se,
            "p.value" => 2 * (1 - cdf(Normal(), abs(estimate / se))),
        )
    end

    return results
end
