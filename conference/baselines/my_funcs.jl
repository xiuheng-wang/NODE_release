eye(n) = Diagonal{Float64}(I, n)

norm2(x) = norm(x)^2




"""
    esp_exp_quad(k,a,b,w,s)

    returns E[exp(s*(y'wy + b'y))]
    where y : N(a,k)
"""
function esp_exp_quad(k,a,b,w,s)
    n = length(a)

    res = exp(s*(dot(a,w*a)+dot(b,a)))
    res /= sqrt(det(I - 2s*w*k))
    res *= exp((s^2/2)*(dot(2*w*a+b, (k*inv(I - 2*s*w*k)*(2*w*a+b)))))

end

"""
    comp_H_h(dict_x, m, R, γ)

    computes matrices H and h
    - dict_x : dictionnary
    - m : mean vector of the signal
    - R : covariance matrix of the signal
    - γ : kernel bandwidth

"""
function comp_H_h(dict_x, m, R, γ)
    (k, l) = size(dict_x)

    h = zeros(l)
    for q in 1:l
        b = -2*dict_x[:,q]

        h[q] = exp(-norm2(dict_x[:,q])/(2*γ))
        h[q] *= esp_exp_quad(R, m, b, eye(k),-1/(2*γ))
    end

    H = zeros(l,l)
    for q in 1:l, n in 1:l
        b = - (dict_x[:,q] + dict_x[:,n])

        H[q,n] = 1exp(-(norm2(dict_x[:,q])+norm2(dict_x[:,n]))/(2*γ))
        H[q,n] *= esp_exp_quad(R, m, b, eye(k), -1/γ)
    end


    return H, h
end


"""
    comp_Γ(dict_x, m, R, γ)

    computes matrix Γ
    - dict_x : dictionnary
    - m : mean vector of the signal
    - R : covariance matrix of the signal
    - γ : kernel bandwidth

"""
function comp_Γ(dict_x, m, R, γ)
    (k, l) = size(dict_x)

    #show(STDOUT, "text/plain", dict_x)
    Γ = zeros(l^2, l^2)

    for q in 1:l, n in 1:l
        for i in 1:l, j in 1:l
            # Γ[(q-1)*l + i, (n-1)*l + j] =E[k_q k_n k_i k_j]

            b = - 2*(dict_x[:,q] + dict_x[:,n] + dict_x[:,i] + dict_x[:,j])

            Γ[(q-1)*l + i, (n-1)*l + j] = 1exp(-(norm2(dict_x[:,q])+norm2(dict_x[:,n])+norm2(dict_x[:,i])+norm2(dict_x[:,j]))/(2*γ))
            Γ[(q-1)*l + i, (n-1)*l + j] *= esp_exp_quad(R, m, b, 4*eye(k), -1/2γ)
        end
    end

    return Γ

end

"""
    comp_Δ(dict_x, m, R, γ)

    computes matrix Δ
    - dict_x : dictionnary
    - m : mean vector of the signal
    - R : covariance matrix of the signal
    - γ : kernel bandwidth

"""
function comp_Δ(dict_x, m, R, γ)
    (k, l) = size(dict_x)

    #show(STDOUT, "text/plain", dict_x)
    Δ = zeros(l^2, l)

    for q in 1:l
        for i in 1:l, j in 1:l
            # Γ[(q-1)*l + i, (n-1)*l + j] =E[k_q k_n k_i k_j]

            b = - 2*(dict_x[:,q] + dict_x[:,i] + dict_x[:,j])

            Δ[(q-1)*l + i, j] = 1exp(-(norm2(dict_x[:,q])+norm2(dict_x[:,i])+norm2(dict_x[:,j]))/(2*γ))
            Δ[(q-1)*l + i, j] *= esp_exp_quad(R, m, b, 3*eye(k), -1/2γ)
        end
    end

    return Δ

end

"""
    comp_κ(x, dict_x, γ; bias = false)

    computes the vector of k(x, x_i^{dict})
"""
function comp_κ(x, dict_x, γ)
    (k, l) = size(dict_x)
    κ = [exp(-norm(x-dict_x[:,m])^2/(2γ)) for m in 1:l]
end


"""
    comp_dict(x, γ, eta)

    computes the dictionary by scanning all the time series and keeping incoherent samples
"""
function comp_dict(x, γ, eta)

    (k, nt) = size(x)
    dict_x = []
    
    for t in 1:nt
        if t == 1
            dict_x = x[:,t]
            dict_x = reshape(dict_x, (k,1)) 
            κ = 10000
        else
            κ = comp_κ(x[:,t], dict_x, γ)
        end
        
        if all(κ .< eta)
            dict_x = hcat(dict_x, x[:,t])
        end
    end
    show(size(dict_x))

    return dict_x
end






"""
    nougat(x, dict, n_ref, n_test, μ, ν, γ)

    computes nougat
"""
function nougat(x, dict, n_ref, n_test, μ, ν, γ)

    (k,l) = size(dict)
    n_iter = size(x, 2) - n_ref - n_test
    nougat = zeros(n_iter)
    θ = zeros(l)

    # # compute initial H_n and h_n

    H_nr = zeros(l,l)
    h_nr = zeros(l)
    for m in 1:n_ref
        κ = comp_κ(x[:,m], dict, γ)
        H_nr += κ*κ'
        h_nr += κ
    end
    H_nr *= 1/n_ref
    h_nr *= 1/n_ref

    h_nt = zeros(l)
    for m in n_ref + 1:n_ref + n_test
        κ = comp_κ(x[:,m], dict, γ)
        h_nt += κ
    end
    h_nt *= 1/n_test

    for n in 1: n_iter-1

        θ = θ - μ*((H_nr + ν*I)*θ + (h_nr - h_nt))

        κ_n = comp_κ(x[:, n_ref+n_test + n], dict, γ)
        nougat[n] =  dot(h_nt , θ)

        # update H_nr, h_nr  and h_nt the fast way

        um_r = comp_κ(x[:, n], dict, γ)
        up_r = comp_κ(x[:, n+n_ref], dict, γ)

        H_nr += (up_r*up_r' - um_r*um_r')/n_ref
        h_nr += (up_r - um_r)/n_ref

        um_t = comp_κ(x[:, n+n_ref], dict, γ)
        up_t = comp_κ(x[:, n+n_ref+n_test], dict, γ)
        h_nt += ( up_t - um_t)/n_test

    end

    θ = θ - μ*((H_nr + ν*I)*θ + (h_nr - h_nt))
    nougat[n_iter] = dot(θ, h_nt)

    return nougat
end

function rulsif(x, dict, n_ref, n_test, ν, γ)

    (k,l) = size(dict)
    n_iter = size(x, 2) - n_ref - n_test
    rulsif = zeros(n_iter)

    # # compute initial h_n

    H_nr = zeros(l,l)
    h_nr = zeros(l)
    for m in 1:n_ref
        κ = comp_κ(x[:,m], dict, γ)
        H_nr += κ*κ'
        h_nr += κ
    end
    H_nr *= 1/n_ref
    h_nr *= 1/n_ref

    h_nt = zeros(l)
    for m in n_ref + 1:n_ref + n_test
        κ = comp_κ(x[:,m], dict, γ)
        h_nt += κ
    end
    h_nt *= 1/n_test


    for n in 1 : n_iter

        θ = (H_nr + ν*I)\(h_nt -  h_nr)
        rulsif[n] =  dot(θ,h_nt)

        # update h_nr  and h_nt the fast way

        um_r = comp_κ(x[:, n], dict, γ)
        up_r = comp_κ(x[:, n+n_ref], dict, γ)
        H_nr += (up_r*up_r' - um_r*um_r')/n_ref
        h_nr += (up_r - um_r)/n_ref

        um_t = comp_κ(x[:, n+n_ref], dict, γ)
        up_t = comp_κ(x[:, n+n_ref+n_test], dict, γ)
        h_nt += ( up_t - um_t)/n_test

    end

    return rulsif
end

function newma(y, dict, n_ref, n_test, γ)
    z = z_p = comp_κ(y[:,1], dict, γ)
    test = Float64[]

    for n in 2:size(y, 2)
        Ψ = comp_κ(y[:,n], dict, γ)
        z = (1-λ)*z + λ*Ψ
        z_p = (1-Λ)*z_p + Λ*Ψ
        push!(test, norm(z - z_p))
    end

    return test

end

"""
    knnt(x, n_ref, n_test, k)

    computes knn test
"""
function knnt(x, n_ref, n_test, k)

    n_s = size(x,2)
    n_iter = n_s - n_ref - n_test + 1

    knn_test = zeros(n_iter)

    for n in 1:n_iter
        data = x[:, n:n+n_ref+n_test-1]
        kdtree = KDTree(data)
        idxs, dists = knn(kdtree, data, k)
        knn_test[n] = sum([sum(idxs[k] .> n_ref) for k in 1:n_ref])
        + sum([sum(idxs[k] .<= n_ref) for k in n_ref+1:n_ref+n_test])
    end

    mean_test = k*n_ref*n_test/(n_ref + n_test - 1)
    return mean_test .- knn_test

end


"""
    ma(x, dict, n_ref, n_test, γ)

    computes ma
"""
function ma(x, dict, n_ref, n_test, γ)

    (k,l) = size(dict)
    n_iter = size(x, 2) - n_ref - n_test
    ma = zeros(n_iter + 1)

    # # compute initial h_n

    h_nr = zeros(l)
    for m in 1:n_ref
        κ = comp_κ(x[:,m], dict, γ)
        h_nr += κ
    end
    h_nr *= 1/n_ref

    h_nt = zeros(l)
    for m in n_ref + 1:n_ref + n_test
        κ = comp_κ(x[:,m], dict, γ)
        h_nt += κ
    end
    h_nt *= 1/n_test

    ma[1] = norm(h_nr - h_nt)

    for n in 1 : n_iter

        # update h_nr  and h_nt the fast way

        um_r = comp_κ(x[:, n], dict, γ)
        up_r = comp_κ(x[:, n+n_ref], dict, γ)
        h_nr += (up_r - um_r)/n_ref

        um_t = comp_κ(x[:, n+n_ref], dict, γ)
        up_t = comp_κ(x[:, n+n_ref+n_test], dict, γ)
        h_nt += ( up_t - um_t)/n_test

        ma[n+1] = norm(h_nr - h_nt)

    end
    return ma
end

function comp_roc(t_0, t_1; n_ξ=128, ξ_pl = nothing)

    if ξ_pl == nothing
        ξ_pl_inf = maximum(minimum(abs.(t_0), dims=1))
        ξ_pl_sup = minimum(maximum(abs.(t_1), dims=1))
        ξ_pl = range(ξ_pl_inf, stop = ξ_pl_sup, length=n_ξ)
    end

    #pfa = [mean(any(x->x>ξ, abs.(t_0), dims=1)) for ξ in ξ_pl]
    pfa = [count(x->x>ξ, abs.(t_0)) for ξ in ξ_pl]/prod(size(t_0))
    pd = [mean(any(x->x>ξ, abs.(t_1), dims=1)) for ξ in ξ_pl]
    #pd = [count(x->x>ξ, abs.(t_1)) for ξ in ξ_pl]/prod(size(t_1))

    return pfa, pd, ξ_pl
end


function comp_pfa(t_0; n_ξ=128, ξ_pl = nothing)

    if ξ_pl == nothing
        ξ_pl_inf = minimum(abs.(t_0))
        ξ_pl_sup = maximum(abs.(t_0))
        ξ_pl = range(ξ_pl_inf, stop = ξ_pl_sup, length=n_ξ)
    end

    pfa = [count(x->x>ξ, abs.(t_0)) for ξ in ξ_pl]/prod(size(t_0))
    # pfa = [mean(any(x->x>ξ, abs.(t_0), dims=1)) for ξ in ξ_pl]
    # pfa = [count(x->x>ξ, abs.(t_0[140, :])) for ξ in ξ_pl]/size(t_0, 2)

    return pfa, ξ_pl
end

"""
    comp_mtd(t; n_ξ=128, ξ_pl = nothing)

    mean of first detection

"""
function comp_mtd(t; n_ξ=128, ξ_pl = nothing)

    if ξ_pl == nothing
        ξ_pl_inf = minimum(abs.(t))
        ξ_pl_sup = maximum(abs.(t))
        ξ_pl = range(ξ_pl_inf, stop = ξ_pl_sup, length=n_ξ)
    end

    mean_delay = Vector{Union{Float64, Missing}}(undef, length(ξ_pl))
    for q in 1:length(ξ_pl)
        first_detec = [findfirst(x -> x > ξ_pl[q], abs.(t[:,k])) for k in 1:size(t, 2)]
        if eltype(first_detec) == Nothing
            mean_delay[q] = missing
        else
            first_detec = filter(!isnothing, first_detec)
            mean_delay[q] = mean(first_detec)
        end
    end

    return mean_delay, ξ_pl
end


"""
    pdf_gmm(d, k; ν = 30.0, σ = 1.0, α = 5.0)

    pdf of a GMM
    k: number of components
    d: dimension

"""
function pdf_gmm(d, k; n = d+2, σ = 1.0, α = 5.0)
    pdf_cov = Wishart(n, Matrix{Float64}(I, d, d))
    pdf_mean = MvNormal(zeros(d), (σ^2)*Matrix{Float64}(I, d, d))
    param_gmm = [(rand(pdf_mean), rand(pdf_cov)/n) for n in 1:k]
    MixtureModel(MvNormal, param_gmm, rand(Dirichlet(α*ones(k))))
end

function median_trick(data)
    n = size(data,2)
    dists = Float64[]
    for j in 1:n, i in 1:j-1
        push!(dists, norm(data[:,j]-data[:,i]))
    end
     median(dists)^2
end
