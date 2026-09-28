# The mediation DGP used by the R and Julia books (from the medoutcon/crumble
# examples): W1, W2, W3 baseline; A treatment; Z a mediator-outcome confounder
# affected by A; M mediator; Y binary outcome. All variables are binary, so every
# functional can be computed exactly by enumeration.
using DataFrames, Distributions, Random, StatsFuns

pw1 = 0.6; pw2 = 0.3
pw3(w1, w2) = min(0.2 + (w1 + w2) / 3, 1.0)
pa(w1, w2, w3) = logistic(w1 + w2 + w3 - 2)
pz(a, w1, w2, w3) = logistic(-log(2) - a + (w1 + w2 + w3) / 3 + 0.2)
pm(a, z, w1, w2) = logistic(log(3) * (w1 + w2) + 2a - 2z)
py(a, z, m, w1, w2, w3) = logistic(1 / (w1 + w2 + w3 - z + a + m))

bern(p, x) = x == 1 ? p : 1 - p

function make_example_data(n; rng = Random.default_rng())
    w1 = rand(rng, n) .< pw1
    w2 = rand(rng, n) .< pw2
    w3 = rand(rng, n) .< pw3.(w1, w2)
    a = rand(rng, n) .< pa.(w1, w2, w3)
    z = rand(rng, n) .< pz.(a, w1, w2, w3)
    m = rand(rng, n) .< pm.(a, z, w1, w2)
    y = rand(rng, n) .< py.(a, z, m, w1, w2, w3)
    DataFrame(W_1 = Int.(w1), W_2 = Int.(w2), W_3 = Int.(w3), A = Int.(a), Z = Int.(z), M = Int.(m), Y = Int.(y))
end

# E_W sum_z p(z|l,W) sum_m p(m|k,W,z) mu(j,W,z,m)
function truth_natural(j, k, l)
    s = 0.0
    for w1 in 0:1, w2 in 0:1, w3 in 0:1
        pwt = bern(pw1, w1) * bern(pw2, w2) * bern(pw3(w1, w2), w3)
        for z in 0:1, m in 0:1
            s += pwt * bern(pz(l, w1, w2, w3), z) * bern(pm(k, z, w1, w2), m) * py(j, z, m, w1, w2, w3)
        end
    end
    s
end

# E_W sum_z' p(z'|j,W) sum_z p(z|l,W) sum_m p(m|k,W,z) mu(i,W,z',m)
function truth_randomized(i, j, k, l)
    s = 0.0
    for w1 in 0:1, w2 in 0:1, w3 in 0:1
        pwt = bern(pw1, w1) * bern(pw2, w2) * bern(pw3(w1, w2), w3)
        for zp in 0:1, z in 0:1, m in 0:1
            s += pwt * bern(pz(j, w1, w2, w3), zp) * bern(pz(l, w1, w2, w3), z) *
                 bern(pm(k, z, w1, w2), m) * py(i, zp, m, w1, w2, w3)
        end
    end
    s
end

function truth()
    n = truth_natural; r = truth_randomized
    Dict(
        "RI_ride" => r(1,1,0,0) - r(0,0,0,0), "RI_riie" => r(1,1,1,1) - r(1,1,0,0),
        "RT_ate" => n(1,1,1) - n(0,0,0),
        "RT_p1" => n(1,1,1) - n(0,1,1), "RT_p2" => r(0,1,1,1) - r(0,0,1,1),
        "RT_p3" => r(0,0,1,1) - r(0,0,1,0), "RT_p4" => n(0,1,0) - n(0,0,0),
        "RT_direct" => n(1,1,1) - n(0,1,1) + r(0,1,1,1) - r(0,0,1,1),
        "RT_indirect" => r(0,0,1,1) - r(0,0,1,0) + n(0,1,0) - n(0,0,0),
    )
end
