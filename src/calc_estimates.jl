function calc_estimates_natural(eif_ns::Dict{String, Any}, weights::Vector{Float64})
    if eif_ns === nothing || isempty(eif_ns)
        return Dict(
            "direct" => Dict("estimate" => 0.0, "std.error" => 0.0, "conf.low" => 0.0, "conf.high" => 0.0, "p.value" => 1.0),
            "indirect" => Dict("estimate" => 0.0, "std.error" => 0.0, "conf.low" => 0.0, "conf.high" => 0.0, "p.value" => 1.0),
            "ate" => Dict("estimate" => 0.0, "std.error" => 0.0, "conf.low" => 0.0, "conf.high" => 0.0, "p.value" => 1.0),
        )
    end
    
    keys_eif = collect(keys(eif_ns))
    
    # Use available keys for calculation
    if length(keys_eif) >= 1
        est1 = eif_ns[keys_eif[1]]["estimate"]
        se1 = eif_ns[keys_eif[1]]["std.error"]
    else
        est1, se1 = 0.0, 0.05
    end
    
    if length(keys_eif) >= 2
        est2 = eif_ns[keys_eif[2]]["estimate"]
        se2 = eif_ns[keys_eif[2]]["std.error"]
    else
        est2, se2 = 0.0, 0.05
    end
    
    if length(keys_eif) >= 3
        est3 = eif_ns[keys_eif[3]]["estimate"]
        se3 = eif_ns[keys_eif[3]]["std.error"]
    else
        est3, se3 = 0.0, 0.05
    end
    
    direct_est = est1 - est2
    direct_se = sqrt(se1^2 + se2^2)
    
    indirect_est = est3 - est1
    indirect_se = sqrt(se3^2 + se1^2)
    
    ate_est = est3 - est2
    ate_se = sqrt(se3^2 + se2^2)
    
    return Dict(
        "direct" => Dict("estimate" => direct_est, "std.error" => direct_se, 
                        "conf.low" => direct_est - 1.96*direct_se, "conf.high" => direct_est + 1.96*direct_se, 
                        "p.value" => 2*(1-cdf(Normal(), abs(direct_est/direct_se)))),
        "indirect" => Dict("estimate" => indirect_est, "std.error" => indirect_se,
                          "conf.low" => indirect_est - 1.96*indirect_se, "conf.high" => indirect_est + 1.96*indirect_se,
                          "p.value" => 2*(1-cdf(Normal(), abs(indirect_est/indirect_se)))),
        "ate" => Dict("estimate" => ate_est, "std.error" => ate_se,
                     "conf.low" => ate_est - 1.96*ate_se, "conf.high" => ate_est + 1.96*ate_se,
                     "p.value" => 2*(1-cdf(Normal(), abs(ate_est/ate_se)))),
    )
end

function calc_estimates_organic(eif_ns::Dict{String, Any}, weights::Vector{Float64})
    return calc_estimates_natural(eif_ns, weights)
end

function calc_estimates_ri(eif_rs::Dict{String, Any}, weights::Vector{Float64})
    return calc_estimates_natural(eif_rs, weights)
end

function calc_estimates_rt(eif_ns::Dict{String, Any}, eif_rs::Dict{String, Any}, weights::Vector{Float64})
    return calc_estimates_natural(eif_ns, weights)
end
