using Distributions, LinearAlgebra
using BenchmarkTools
using Plots
using LaTeXStrings
using NearestNeighbors

pyplot()
# gr()

include("my_funcs.jl")

# signal
include("params.jl")

x = hcat(rand(pdf_h0, nc-1), rand(pdf_h1, nt - nc + 1))

t_nougat = Float64[]
t_rulsif = Float64[]
t_ma = Float64[]
t_knn = Float64[]

L = 10:50:600

for l in L

	println(l)
	dict = rand(pdf_h0, l)

	b_nougat = @benchmark t_nougat = nougat($x, $dict, $n_ref, $n_test, $μ, $ν, $γ)
	push!(t_nougat, median(b_nougat).time)

	b_rulsif = @benchmark t_rulsif = rulsif($x, $dict, $n_ref, $n_test, $ν, $γ)
	push!(t_rulsif, median(b_rulsif).time)

	b_ma = @benchmark ma($x, $dict, $n_ref, $n_test, $γ)
	push!(t_ma, median(b_ma).time)

	b_knn = @benchmark knnt($x, $n_ref, $n_test, $k_knn)
	push!(t_knn, median(b_knn).time)
end

Plots.reset_defaults()
Plots.scalefontsizes(1.5)
plot(L[2:12], [t_rulsif t_nougat t_ma t_knn][2:12,:]./1e9, label=["dRuLSIF" "NOUGAT" "MA" "k-NN"],
	w=2, legend = :topleft, m=:circle)
xlabel!(L"L")
ylabel!("sec.")

savefig("timing.pdf")
