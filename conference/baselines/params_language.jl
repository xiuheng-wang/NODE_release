# data

d = 20
k = 9

pdf_h0 = pdf_gmm(d, k; σ = 1.3)
pdf_h1 = pdf_gmm(d, k; σ = 1.3)

# cpd
nt = 2144

# nougat

n_ref = n_test = 64
μ = 0.1   # step size
ν = 0.005    # ridge regularization
dict = [rand(pdf_h0, 6) rand(pdf_h1, 6)]

# kernel bandwidth

data = rand(pdf_h0, 100)
γ = median_trick(data)

# knn
k_knn = 10
