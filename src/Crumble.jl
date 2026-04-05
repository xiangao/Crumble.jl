module Crumble

using DataFrames
using Flux
using Optimisers
using MLJ
using MLJBase
using MLJLinearModels
using MLJModels
using Statistics
using LinearAlgebra
using JuMP
using HiGHS
using ProgressMeter
using StatsBase
using CategoricalArrays
using Random
using Tables
using OneHotArrays

export crumble, crumble_control, sequential_module, CrumbleResult

include("types.jl")
include("helpers.jl")
include("assertions.jl")
include("shift.jl")
include("params.jl")
include("dataset.jl")
include("alpha.jl")
include("theta.jl")
include("phi.jl")
include("eif.jl")
include("calc_estimates.jl")
include("permutation.jl")
include("main.jl")
include("display.jl")

end # module
