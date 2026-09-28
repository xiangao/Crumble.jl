# Uncentered efficient influence functions (R crumble's eif_n / eif_r). The
# estimate of each functional is the (weighted) mean of its influence function.
#
#   natural:     alpha3 (Y - fit3) + alpha2 (b3 - fit2) + alpha1 (b2 - fit1) + b1
#   randomized:  alpha4 (Y - fit4) + alpha3 (b4 - fit3) + alpha2 (b3 - fit2)
#                + alpha1 (b2 - fit1) + b1
#
# Observations with an unobserved outcome contribute no outcome residual; their
# alpha3 (alpha4) is zero in large samples because C = 1 is part of the shift.

function outcome_residual(cd::CrumbleData, fit)
    Y = cd.data[:, cd.vars.Y]
    observed = .!ismissing.(Y)
    if cd.vars.C !== nothing
        observed .&= cd.data[:, cd.vars.C] .== 1
    end
    [observed[i] ? Float64(Y[i]) - fit[i] : 0.0 for i in eachindex(Y)]
end

function eif_n(cd, th, a)
    a["alpha3"] .* outcome_residual(cd, th["fit3"]) .+
    a["alpha2"] .* (th["b3"] .- th["fit2"]) .+
    a["alpha1"] .* (th["b2"] .- th["fit1"]) .+
    th["b1"]
end

function eif_r(cd, th, a)
    a["alpha4"] .* outcome_residual(cd, th["fit4"]) .+
    a["alpha3"] .* (th["b4"] .- th["fit3"]) .+
    a["alpha2"] .* (th["b3"] .- th["fit2"]) .+
    a["alpha1"] .* (th["b2"] .- th["fit1"]) .+
    th["b1"]
end

# An estimate carried together with its observation-level influence function, so
# that contrasts of functionals estimated on the same sample get the covariance
# right: the SE of a difference is computed from the difference of the IFs.
struct IFEstimate
    estimate::Float64
    eif::Vector{Float64}
    weights::Vector{Float64}
    id::Vector{Int}
end

function IFEstimate(eif::Vector{Float64}, weights::Vector{Float64}, id::Vector{Int})
    all(isfinite, eif) || error("Crumble.jl: non-finite influence function values; check positivity " *
                                "(extreme Riesz representers) or the outcome regressions.")
    IFEstimate(sum(weights .* eif) / sum(weights), eif, weights, id)
end

Base.:-(a::IFEstimate, b::IFEstimate) = IFEstimate(a.eif .- b.eif, a.weights, a.id)
Base.:+(a::IFEstimate, b::IFEstimate) = IFEstimate(a.eif .+ b.eif, a.weights, a.id)

# Cluster-robust IF variance: sum the weighted, centered IF within clusters, then
# se = sqrt(G/(G-1) * sum_g S_g^2) / n. Without clusters (G = n) and unit weights
# this is sd(IF)/sqrt(n).
function std_error(e::IFEstimate)
    n = length(e.eif)
    c = e.weights .* (e.eif .- e.estimate)
    sums = Dict{Int, Float64}()
    for i in 1:n
        sums[e.id[i]] = get(sums, e.id[i], 0.0) + c[i]
    end
    G = length(sums)
    G < 2 && return NaN
    sqrt(G / (G - 1) * sum(abs2, values(sums))) / n
end

function summarize(e::IFEstimate)
    se = std_error(e)
    degenerate = !isfinite(se) || se < 1e-10
    se = degenerate ? NaN : se
    Dict{String, Any}(
        "estimate" => e.estimate,
        "std.error" => se,
        "conf.low" => e.estimate - 1.96 * se,
        "conf.high" => e.estimate + 1.96 * se,
        "p.value" => degenerate ? NaN : 2 * (1 - cdf(Normal(), abs(e.estimate / se))),
        "influence" => e.eif,
    )
end

function calc_eifs(cd::CrumbleData, alphas, thetas, family::String, f)
    alphas === nothing && return nothing
    id = cluster_ids(cd)
    Dict{String, IFEstimate}(key => IFEstimate(f(cd, thetas[family][key], a), cd.weights, id)
                             for (key, a) in alphas)
end
