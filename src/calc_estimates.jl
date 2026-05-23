function calc_estimates_natural(eif_ns::Dict{String, Any}, weights::Vector{Float64})
    if eif_ns === nothing || isempty(eif_ns)
        return Dict(
            "direct" => Dict("estimate" => 0.0, "std.error" => 0.0, "conf.low" => 0.0, "conf.high" => 0.0, "p.value" => 1.0),
            "indirect" => Dict("estimate" => 0.0, "std.error" => 0.0, "conf.low" => 0.0, "conf.high" => 0.0, "p.value" => 1.0),
            "ate" => Dict("estimate" => 0.0, "std.error" => 0.0, "conf.low" => 0.0, "conf.high" => 0.0, "p.value" => 1.0),
        )
    end
    
    # Retrieve estimates using explicit keys to avoid unstable Dict key order
    haskey_all = haskey(eif_ns, "100") && haskey(eif_ns, "000") && haskey(eif_ns, "111")
    
    key_100 = haskey_all ? "100" : collect(keys(eif_ns))[1]
    key_000 = haskey_all ? "000" : (length(keys(eif_ns)) >= 2 ? collect(keys(eif_ns))[2] : key_100)
    key_111 = haskey_all ? "111" : (length(keys(eif_ns)) >= 3 ? collect(keys(eif_ns))[3] : key_000)
    
    est_100 = eif_ns[key_100]["estimate"]
    se_100  = eif_ns[key_100]["std.error"]
    
    est_000 = eif_ns[key_000]["estimate"]
    se_000  = eif_ns[key_000]["std.error"]
    
    est_111 = eif_ns[key_111]["estimate"]
    se_111  = eif_ns[key_111]["std.error"]
    
    direct_est = est_100 - est_000
    direct_se = sqrt(se_100^2 + se_000^2)
    
    indirect_est = est_111 - est_100
    indirect_se = sqrt(se_111^2 + se_100^2)
    
    ate_est = est_111 - est_000
    ate_se = sqrt(se_111^2 + se_000^2)
    
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
