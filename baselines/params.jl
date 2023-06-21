# data

d = 6
k = 3

pdf_h0 = pdf_gmm(d, k; σ = 1)
pdf_h1 = pdf_gmm(d, k; σ = 1)

# cpd
nc = 400
nt = 700

# nougat

n_ref = n_test = 64
μ = 0.047   # step size
ν = 0.01    # ridge regularization
dict = [rand(pdf_h0, 40) rand(pdf_h1, 40)]

# kernel bandwidth

data = rand(pdf_h0, 100)
γ = median_trick(data)

# knn
k_knn = 10
