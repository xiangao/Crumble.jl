# True sequential Riesz representers at each observation (randomized functionals).
pwf(w1, w2, w3) = bern(pw1, w1) * bern(pw2, w2) * bern(pw3(w1, w2), w3)
pawf(a, w...) = pwf(w...) * bern(pa(w...), a)
pazwf(a, z, w...) = pawf(a, w...) * bern(pz(a, w...), z)
pazmwf(a, z, m, w...) = pazwf(a, z, w...) * bern(pm(a, z, w[1], w[2]), m)
pamwf(a, m, w...) = sum(pazmwf(a, z, m, w...) for z in 0:1)
function true_alpha_r(d, i, j, k, l, stage)
    a1(a, w...) = a == l ? pwf(w...) / pawf(l, w...) : 0.0
    a2(a, z, w...) = a == k ? sum(a1(b, w...) * pazwf(b, z, w...) for b in 0:1) / pazwf(k, z, w...) : 0.0
    a3(a, m, w...) = a == j ? sum(a2(b, z, w...) * pazmwf(b, z, m, w...) for b in 0:1, z in 0:1) / pamwf(j, m, w...) : 0.0
    a4(a, z, m, w...) = a == i ? sum(a3(b, m, w...) * pamwf(b, m, w...) * bern(pz(b, w...), z) for b in 0:1) / pazmwf(i, z, m, w...) : 0.0
    [begin
        w = (r.W_1, r.W_2, r.W_3)
        stage == "alpha3" ? a3(r.A, r.M, w...) : a4(r.A, r.Z, r.M, w...)
     end for r in eachrow(d)]
end
