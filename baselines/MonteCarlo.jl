using Plots, Distributions, LinearAlgebra
using NearestNeighbors
using NPZ

include("my_funcs.jl")
pyplot()
# signal

include("params.jl")

# comp test statistics

include("MonteCarlo_distrib.jl")
