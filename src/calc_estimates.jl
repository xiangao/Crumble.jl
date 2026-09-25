"""
    contrast(eif_a, eif_b, w)

Form the contrast of two influence-function based estimates computed on the same
sample. The two estimates share observations, so the covariance term is part of
the variance of their difference. Using `sqrt(se_a^2 + se_b^2)` assumes
independence and is wrong here: it overstates the SE when the two influence
curves are positively correlated, which is the usual case.

The correct quantity is the SD of the observation-level contrast influence
function, `IF_a - IF_b`, divided by `sqrt(n)`.
"""
function contrast(eif_a::Vector{Float64}, eif_b::Vector{Float64}, w::Vector{Float64})
    length(eif_a) == length(eif_b) || error("Crumble.jl: contrast influence functions have different lengths.")
    d = eif_a .- eif_b
    est = sum(d .* w) / sum(w)
    n = length(d)
    se = std(d) / sqrt(n)
    degenerate = !isfinite(se) || se < 1e-10
    return Dict(
        "estimate"   => est,
        "std.error"  => degenerate ? NaN : se,
        "conf.low"   => degenerate ? NaN : est - 1.96 * se,
        "conf.high"  => degenerate ? NaN : est + 1.96 * se,
        "p.value"    => degenerate ? NaN : 2 * (1 - cdf(Normal(), abs(est / se))),
    )
end

function calc_estimates_natural(eif_ns::Dict{String, Any}, weights::Vector{Float64})
    if eif_ns === nothing || isempty(eif_ns)
        # QUARANTINED: this returned zeros with zero SEs and p = 1, which reads as
        # a precisely estimated null rather than as a failure.
        error("Crumble.jl: calc_estimates_natural received no influence functions.")
    end

    # The three functionals are required by name. Substituting whichever keys
    # happen to be present, as the previous code did, silently reports one
    # estimand under another estimand's label.
    for k in ("100", "000", "111")
        haskey(eif_ns, k) || error("Crumble.jl: required functional \"$k\" is missing; " *
                                   "cannot form the natural-effect contrasts.")
    end

    getif(k) = haskey(eif_ns[k], "influence") ? Vector{Float64}(eif_ns[k]["influence"]) :
        error("Crumble.jl: functional \"$k\" carries no observation-level influence " *
              "function, so a valid contrast SE cannot be formed.")

    if_100, if_000, if_111 = getif("100"), getif("000"), getif("111")

    return Dict(
        "direct"   => contrast(if_100, if_000, weights),
        "indirect" => contrast(if_111, if_100, weights),
        "ate"      => contrast(if_111, if_000, weights),
    )
end

# The organic, randomized-interventional and randomized-transported estimands are
# distinct functionals. They previously all delegated to the natural-effect
# calculation, so three different labels reported the same numbers.
function calc_estimates_organic(eif_ns::Dict{String, Any}, weights::Vector{Float64})
    error("Crumble.jl: organic effects are not implemented; this previously " *
          "returned the natural-effect estimates under an organic label.")
end

function calc_estimates_ri(eif_rs::Dict{String, Any}, weights::Vector{Float64})
    error("Crumble.jl: randomized-interventional effects are not implemented; this " *
          "previously returned the natural-effect estimates under an RI label.")
end

function calc_estimates_rt(eif_ns::Dict{String, Any}, eif_rs::Dict{String, Any}, weights::Vector{Float64})
    error("Crumble.jl: randomized-transported effects are not implemented; this " *
          "previously returned the natural-effect estimates under an RT label.")
end
