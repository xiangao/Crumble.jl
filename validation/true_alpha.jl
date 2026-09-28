# Exact sequential Riesz representers for the randomized functional (i,j,k,l)
# under the book DGP, by enumeration. Shows how large the fourth-stage weight is.
include("dgp.jl")
using Printf
W = [(w1, w2, w3) for w1 in 0:1 for w2 in 0:1 for w3 in 0:1]
pw(w) = bern(pw1, w[1]) * bern(pw2, w[2]) * bern(pw3(w[1], w[2]), w[3])
paw(a, w) = pw(w) * bern(pa(w...), a)
pazw(a, z, w) = paw(a, w) * bern(pz(a, w...), z)
pazmw(a, z, m, w) = pazw(a, z, w) * bern(pm(a, z, w[1], w[2]), m)
pamw(a, m, w) = sum(pazmw(a, z, m, w) for z in 0:1)
function alphas(i, j, k, l)
    a1(a, w) = a == l ? pw(w) / paw(l, w) : 0.0
    a2(a, z, w) = a == k ? sum(a1(a′, w) * pazw(a′, z, w) for a′ in 0:1) / pazw(k, z, w) : 0.0
    a3(a, m, w) = a == j ? sum(a2(a′, z, w) * pazmw(a′, z, m, w) for a′ in 0:1, z in 0:1) / pamw(j, m, w) : 0.0
    a4(a, z, m, w) = a == i ? sum(a3(a′, m, w) * pamw(a′, m, w) * bern(pz(a′, w...), z) for a′ in 0:1) / pazmw(i, z, m, w) : 0.0
    cells = [(a, z, m, w) for a in 0:1, z in 0:1, m in 0:1, w in W]
    v4 = [a4(c...) for c in cells]; p = [pazmw(c...) for c in cells]
    (max4 = maximum(v4), mean4 = sum(v4 .* p), sd4 = sqrt(sum(v4 .^ 2 .* p) - sum(v4 .* p)^2),
     argmax = cells[argmax(v4)], pcell = p[argmax(v4)])
end
for key in ((1,1,0,0), (0,0,0,0), (1,1,1,1))
    r = alphas(key...)
    @printf "%s: alpha4 max %.1f at %s (cell prob %.5f; expected obs at n=800: %.2f); E=%.3f sd=%.2f\n" join(key) r.max4 string(r.argmax) r.pcell 800r.pcell r.mean4 r.sd4
end
