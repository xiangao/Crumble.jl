module Crumble

using DataFrames
using Flux
using Optimisers
using Statistics
using LinearAlgebra
using Random
using Distances
using Distributions: Normal, cdf
using Printf

export crumble, crumble_control, sequential_module, CrumbleResult

include("types.jl")
include("helpers.jl")
include("assertions.jl")
include("shift.jl")
include("params.jl")
include("learners.jl")
include("riesz.jl")
include("theta.jl")
include("eif.jl")
include("calc_estimates.jl")
include("permutation.jl")
include("main.jl")
include("display.jl")

end # module
