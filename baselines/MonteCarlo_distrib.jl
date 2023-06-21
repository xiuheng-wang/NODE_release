using SharedArrays
using Distributed
@everywhere using Distributions
@everywhere using LinearAlgebra
@everywhere using NearestNeighbors
@everywhere using NPZ
@everywhere using JLD
@everywhere using HDF5
@everywhere include("my_funcs.jl")

include("params.jl")
# comp test statistics

realmax = 100000

t_nougat = SharedArray{Float64,2}((nt - n_ref - n_test, realmax))
t_rulsif = SharedArray{Float64,2}((nt - n_ref - n_test, realmax))
t_newma =  SharedArray{Float64,2}((nt - n_ref - n_test + 1, realmax))
t_knn =  SharedArray{Float64,2}((nt - n_ref - n_test + 1, realmax))
X =  SharedArray{Float64,3}((d, nt, realmax))

@sync @distributed for k in 1:realmax
    (k % 100 == 0) && println("> ", k)

    x = hcat(rand(pdf_h0, nc - 1), rand(pdf_h1, nt - nc + 1))
    X[:, :, k] = x
    t_nougat[:, k] = nougat(x, dict, n_ref, n_test, μ, ν, γ)
    t_rulsif[:, k] = rulsif(x, dict, n_ref, n_test, ν, γ)
    t_newma[:, k] = ma(x, dict, n_ref, n_test, γ)
    t_knn[:, k] = knnt(x, n_ref, n_test, k_knn)
end
# npzwrite("./data/data.npy", X)
# npzwrite("./data/nougat.npy", t_nougat)
# npzwrite("./data/newma.npy", t_newma)
# npzwrite("./data/knn.npy", t_knn)

# save("./data/data.jld", "data", X)
# save("./data/nougat.jld", "nougat", t_nougat)
# save("./data/newma.jld", "newma", t_newma)
# save("./data/knn.jld", "knn", t_knn)

h5open("./data/data.h5", "w") do file
    write(file, "data", X)
end
h5open("./data/nougat.h5", "w") do file
    write(file, "nougat", t_nougat)
end
h5open("./data/newma.h5", "w") do file
    write(file, "newma", t_newma)
end
h5open("./data/knn.h5", "w") do file
    write(file, "knn", t_knn)
end
