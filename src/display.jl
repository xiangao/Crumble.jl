using Printf

function Base.show(io::IO, result::CrumbleResult)
    println(io, "CrumbleResult")
    println(io, "  Effect type: $(result.effect)")
    println(io, "")
    println(io, "Estimates:")

    effect_names = Dict(
        "direct" => "Direct Effect",
        "indirect" => "Indirect Effect",
        "ate" => "Average Treatment Effect",
        "ode" => "Organic Direct Effect",
        "oie" => "Organic Indirect Effect",
        "ride" => "Randomized Interventional Direct Effect",
        "riie" => "Randomized Interventional Indirect Effect",
        "p1" => "Path: A -> Y",
        "p2" => "Path: A -> Z -> Y",
        "p3" => "Path: A -> Z -> M -> Y",
        "p4" => "Path: A -> M -> Y",
        "intermediate_confounding" => "Intermediate Confounding",
    )

    for (key, val) in result.estimates
        name = get(effect_names, key, key)
        est = val["estimate"]
        se = val["std.error"]
        ci_low = est - 1.96 * se
        ci_high = est + 1.96 * se
        @printf(io, "  %-40s %8.4f (SE: %8.4f) [95%% CI: %8.4f, %8.4f]\n",
                name, est, se, ci_low, ci_high)
    end
end

function tidy(result::CrumbleResult)
    rows = []
    for (estimand, val) in result.estimates
        est = val["estimate"]
        se = val["std.error"]
        ci_low = est - 1.96 * se
        ci_high = est + 1.96 * se
        p_val = 2 * (1 - cdf(Normal(), abs(est / se)))
        push!(rows, (
            estimand=estimand,
            estimate=est,
            std_error=se,
            conf_low=ci_low,
            conf_high=ci_high,
            p_value=p_val,
        ))
    end
    return DataFrame(rows)
end
