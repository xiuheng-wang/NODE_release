using LinearAlgebra
using Distributions
using Plots
pyplot()
using LaTeXStrings

include("my_funcs.jl")
print("hello world!\n")
ν = 1e-3
μ = 5e-4
n_ref = n_test = 250
L = 16
γ = 0.25

mean_sig = zeros(2)
cov_sig = 0.5*[1 0.25; 0.25 1]

print("hello world!\n")
dict_x = rand(MvNormal(mean_sig, cov_sig ), L)
H, h = comp_H_h(dict_x, mean_sig, cov_sig, γ)
Γ = comp_Γ(dict_x, mean_sig, cov_sig, γ)


#μ_pl = range(0, stop = 1/maximum(eigvals(H + ν*I)), length=32)
μ_pl = range(1e-4, stop = 1.0, length=256)
var_∞ = Float64[]
var_∞_approx = Float64[]

print("hello world!\n")
for μ in μ_pl
    print(μ)
    Q = (H - h*h')*(n_ref + n_test)/(n_ref*n_test)

    S = (μ^2/n_ref)*(Γ + (n_ref - 1) * kron(H, H)) + (1-μ*ν)^2*I
    H_plus_H = kron(H,eye(L)) + kron(eye(L), H)
    S -= μ*(1-μ*ν)*H_plus_H


    c_∞ = μ^2*((I - S)\vec(Q))
    push!(var_∞, tr(H*reshape(c_∞, L, L))/n_test)

    push!(var_∞_approx, dot(vec(H),(2*ν*I +H_plus_H)\vec(Q))*μ/n_test)
end

# plots
Plots.reset_defaults()
Plots.scalefontsizes(1.5)
plot(μ_pl , [var_∞, var_∞_approx], w=2, label=["\$var\\{g_\\infty\\}\$ Eqs. (40,41)" "\$var\\{g_\\infty\\}\$  first order approximation Eq. (43)"], legend=:bottomright, yscale = :log10)
xlabel!(L"\mu")

savefig("small_step.pdf")
