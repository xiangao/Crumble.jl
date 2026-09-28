# Effects as contrasts of functionals (R crumble's calc_estimates_*). Keys are the
# treatment patterns of the functionals: "jkl" for natural-type, "ijkl" for
# randomized-type, with 1 = the d1 regime and 0 = the d0 regime.

function need(eifs, keys...)
    eifs === nothing && error("Crumble.jl: no influence functions were computed for this effect.")
    for k in keys
        haskey(eifs, k) || error("Crumble.jl: required functional \"$k\" is missing.")
    end
end

function calc_estimates_natural(n)
    need(n, "111", "100", "000")
    Dict("direct"   => summarize(n["100"] - n["000"]),   # A -> Y
         "indirect" => summarize(n["111"] - n["100"]),   # A -> M -> Y
         "ate"      => summarize(n["111"] - n["000"]))
end

function calc_estimates_organic(n)
    need(n, "101", "000", "111")
    Dict("ode" => summarize(n["101"] - n["000"]),
         "oie" => summarize(n["111"] - n["101"]))
end

# Randomized interventional effects: "1100" is E[Y(1, Z(1), G(0))], with G(0) a
# random draw of the mediator under A = 0 given W, which is the interventional
# direct-effect functional of Diaz et al. (2021) that medoutcon estimates.
function calc_estimates_ri(r)
    need(r, "1100", "0000", "1111")
    Dict("ride" => summarize(r["1100"] - r["0000"]),
         "riie" => summarize(r["1111"] - r["1100"]))
end

# Recanting twins. The path effects p1..p4 and the intermediate-confounding term
# add up to the ATE exactly. (R crumble writes the confounding term with a
# redundant r["0011"] - r["0011"], which is zero; it is omitted here.)
function calc_estimates_rt(n, r)
    need(n, "111", "011", "010", "000")
    need(r, "0111", "0011", "0010")
    p1 = n["111"] - n["011"]         # A -> Y
    p2 = r["0111"] - r["0011"]       # A -> Z -> Y
    p3 = r["0011"] - r["0010"]       # A -> Z -> M -> Y
    p4 = n["010"] - n["000"]         # A -> M -> Y
    ic = n["011"] - r["0111"] + r["0010"] - n["010"]
    Dict("p1" => summarize(p1), "p2" => summarize(p2), "p3" => summarize(p3),
         "p4" => summarize(p4), "intermediate_confounding" => summarize(ic),
         "direct" => summarize(p1 + p2), "indirect" => summarize(p3 + p4),
         "ate" => summarize(n["111"] - n["000"]))
end
