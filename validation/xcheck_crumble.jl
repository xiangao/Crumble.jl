# Crumble.jl RI on the same data as xcheck_medoutcon.R.
using Crumble, CSV, DataFrames, Random, Printf
d = CSV.read(joinpath(@__DIR__, "results", "xcheck_n5000.csv"), DataFrame)
for (learner, riesz) in [("glm", :nn), ("saturated", :nn)]
    Random.seed!(1)
    r = crumble(d, ["A"]; outcome = "Y", mediators = ["M"], moc = ["Z"], covar = ["W_1", "W_2", "W_3"],
                effect = "RI", learners = [learner], control = crumble_control(riesz = riesz))
    for (k, lab) in (("ride", "direct"), ("riie", "indirect"))
        e = r.estimates[k]
        @printf "crumble %-9s %-3s %-8s %.4f (SE %.4f)\n" learner riesz lab e["estimate"] e["std.error"]
    end
end
