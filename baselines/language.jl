using SharedArrays
using Distributed
@everywhere using Distributions
@everywhere using LinearAlgebra
@everywhere using NearestNeighbors
@everywhere using NPZ
@everywhere include("my_funcs.jl")

include("params_language.jl")
# comp test statistics

realmax = 1

t_nougat = SharedArray{Float64,2}((nt - n_ref - n_test, realmax))
t_rulsif = SharedArray{Float64,2}((nt - n_ref - n_test, realmax))
t_newma =  SharedArray{Float64,2}((nt - n_ref - n_test + 1, realmax))
t_knn =  SharedArray{Float64,2}((nt - n_ref - n_test + 1, realmax))

@sync @distributed for k in 1:realmax
    (k % 100 == 0) && println("> ", k)

    x = npzread("./data/Language Detection.npy")
    print(size(x))

    eta = 0.9
    dict = comp_dict(x, γ, eta)

    t_nougat = nougat(x, dict, n_ref, n_test, μ, ν, γ)
    t_rulsif = rulsif(x, dict, n_ref, n_test, ν, γ)
    t_newma = ma(x, dict, n_ref, n_test, γ)
    t_knn = knnt(x, n_ref, n_test, k_knn)
    npzwrite("./data/nougat_language.npy", t_nougat)
    npzwrite("./data/newma_language.npy", t_newma)
    npzwrite("./data/knn_language.npy", t_knn)
end