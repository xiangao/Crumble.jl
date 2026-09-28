# Riesz representers.
#
# Each stage s solves the Riesz regression
#
#     alpha_s = argmin_a  E[ a(X)^2 - 2 w(X) a(X^shift) ],
#
# whose minimiser satisfies E[alpha_s(X) h(X)] = E[w(X) h(X^shift)] for every h.
# X^shift is the same observation with the treatment (or Z') replaced as the
# functional requires, and w is the previous stage's representer evaluated at the
# observed data (w = 1 at the first stage). This is R crumble's
# nn_sequential_riesz_representer.
#
# Two ways to fit it:
#   :nn      — the R package's neural network (ELU MLP, softplus output), Adam with
#              weight decay 0.01 and a one-cycle learning-rate schedule. Unlike the
#              R code, each mini-batch uses the weights of its own observations:
#              R multiplies the full-sample weight vector by the batch output, which
#              broadcasts to an n x batch matrix and so weights every observation
#              by the mean weight.
#   :linear  — a(x) = phi(x)'b over a fixed basis, with the closed-form solution
#              b = (Phi'Phi/n + lambda I)^-1 Phi_shift' w / n. With the :saturated
#              basis (one indicator per covariate pattern) this is the exact
#              empirical Riesz representer for discrete data.

struct RieszBasis
    kind::Symbol
    patterns::Dict{Vector{Float64}, Int}
end

function make_basis(kind::Symbol, Xs::AbstractMatrix...)
    kind in (:main, :saturated) || throw(ArgumentError("Crumble.jl: riesz_basis must be :main or :saturated."))
    patterns = Dict{Vector{Float64}, Int}()
    if kind == :saturated
        for X in Xs, i in axes(X, 1)
            key = Vector{Float64}(X[i, :])
            haskey(patterns, key) || (patterns[key] = length(patterns) + 1)
        end
    end
    RieszBasis(kind, patterns)
end

function basis_matrix(b::RieszBasis, X::AbstractMatrix)
    b.kind == :main && return hcat(ones(size(X, 1)), X)
    Phi = zeros(size(X, 1), length(b.patterns))
    for i in axes(X, 1)
        j = get(b.patterns, Vector{Float64}(X[i, :]), 0)
        j > 0 && (Phi[i, j] = 1.0)
    end
    Phi
end

# The saturated basis is built from the observed training patterns only. A shifted
# row whose pattern never occurs in the observed training data has no in-sample
# representer (a positivity failure); its mass is dropped rather than divided by
# the ridge, and the dropped share is warned about. A validation row with an unseen
# pattern gets alpha = 0 for the same reason.
function riesz_linear(X, Xshift, w, Xvalid, control)
    b = make_basis(control.riesz_basis, X)
    Phi = basis_matrix(b, X)
    Phis = basis_matrix(b, Xshift)
    n = size(X, 1)
    if b.kind == :saturated
        unseen = vec(sum(Phis, dims = 2)) .== 0
        dropped = sum(abs.(w[unseen])) / sum(abs.(w))
        dropped > 0.01 && @warn "Crumble.jl: saturated Riesz basis drops $(round(100dropped, digits = 1))% of the shifted mass (covariate patterns absent from the training data)."
    end
    beta = (Phi' * Phi ./ n + control.riesz_ridge * I) \ (Phis' * w ./ n)
    (train = Phi * beta, valid = basis_matrix(b, Xvalid) * beta)
end

# torch::lr_one_cycle defaults: pct_start 0.3, cosine annealing, div_factor 25,
# final_div_factor 1e4. The R code steps the scheduler once per epoch.
function one_cycle_lr(step::Int, total::Int, max_lr::Float64)
    initial = max_lr / 25
    minimum_lr = initial / 1e4
    end1 = 0.3 * total - 1
    end2 = total - 1
    cosanneal(a, b, pct) = b + (a - b) / 2 * (cos(pi * pct) + 1)
    step <= end1 ? cosanneal(initial, max_lr, end1 <= 0 ? 1.0 : step / end1) :
                   cosanneal(max_lr, minimum_lr, (step - end1) / (end2 - end1))
end

function riesz_nn(X, Xshift, w, Xvalid, control, nn_module)
    xt = Float32.(permutedims(X))
    xs = Float32.(permutedims(Xshift))
    wt = Float32.(w)
    n = size(xt, 2)
    model = nn_module(size(xt, 1))
    rule = Optimisers.OptimiserChain(Optimisers.WeightDecay(0.01f0), Optimisers.Adam(Float32(one_cycle_lr(0, control.epochs, control.learning_rate))))
    state = Optimisers.setup(rule, model)
    Flux.trainmode!(model)
    for epoch in 1:control.epochs
        Optimisers.adjust!(state, Float32(one_cycle_lr(epoch - 1, control.epochs, control.learning_rate)))
        order = randperm(n)
        for start in 1:control.batch_size:n
            idx = order[start:min(start + control.batch_size - 1, n)]
            xb, xsb, wb = xt[:, idx], xs[:, idx], wt[idx]
            grads = Flux.gradient(m -> mean(vec(m(xb)) .^ 2 .- 2 .* wb .* vec(m(xsb))), model)
            Optimisers.update!(state, model, grads[1])
        end
    end
    Flux.testmode!(model)
    (train = Float64.(vec(model(xt))), valid = Float64.(vec(model(Float32.(permutedims(Xvalid))))))
end

function fit_alpha(X, Xshift, w, Xvalid, control, nn_module)
    a = control.riesz == :linear ? riesz_linear(X, Xshift, w, Xvalid, control) :
        control.riesz == :nn ? riesz_nn(X, Xshift, w, Xvalid, control, nn_module) :
        throw(ArgumentError("Crumble.jl: riesz must be :nn or :linear."))
    isfinite(control.alpha_cap) || return a
    (train = min.(a.train, control.alpha_cap), valid = min.(a.valid, control.alpha_cap))
end

colmatrix(df::DataFrame, cols) = Matrix{Float64}(df[:, cols])

# Sequential representers for a natural-type functional (j, k, l):
#   alpha1 on (A, W),          shift to data_l
#   alpha2 on (A, Z, W),       shift to data_k, weighted by alpha1
#   alpha3 on (A, C, M, Z, W), shift to data_j, weighted by alpha2
function phi_n_alpha(train, valid, vars::CrumbleVars, nn_module, p, control)
    j, k, l = Symbol(p["j"]), Symbol(p["k"]), Symbol(p["l"])
    c1 = present(train.data, [vars.A; vars.W])
    c2 = present(train.data, [vars.A; vars.Z; vars.W])
    c3 = present(train.data, [vars.A; cvec(vars.C); vars.M; vars.Z; vars.W])
    a1 = fit_alpha(colmatrix(train.data, c1), colmatrix(train[l], c1), ones(nrow(train.data)), colmatrix(valid.data, c1), control, nn_module)
    a2 = fit_alpha(colmatrix(train.data, c2), colmatrix(train[k], c2), a1.train, colmatrix(valid.data, c2), control, nn_module)
    a3 = fit_alpha(colmatrix(train.data, c3), colmatrix(train[j], c3), a2.train, colmatrix(valid.data, c3), control, nn_module)
    Dict("alpha1" => a1.valid, "alpha2" => a2.valid, "alpha3" => a3.valid)
end

# Sequential representers for a randomized functional (i, j, k, l):
#   alpha1 on (A, W),          shift to data_l
#   alpha2 on (A, Z, W),       shift to data_k, weighted by alpha1
#   alpha3 on (A, M, W),       shift to data_j, weighted by alpha2
#   alpha4 on (A, C, Z, M, W), shift to data_i (Z replaced by Z'), weighted by alpha3
function phi_r_alpha(train, valid, vars::CrumbleVars, nn_module, p, control)
    i, j, k, l = Symbol(p["i"]), Symbol(p["j"]), Symbol(p["k"]), Symbol(p["l"])
    c1 = present(train.data, [vars.A; vars.W])
    c2 = present(train.data, [vars.A; vars.Z; vars.W])
    c3 = present(train.data, [vars.A; vars.M; vars.W])
    c4 = present(train.data, [vars.A; cvec(vars.C); vars.Z; vars.M; vars.W])
    a1 = fit_alpha(colmatrix(train.data, c1), colmatrix(train[l], c1), ones(nrow(train.data)), colmatrix(valid.data, c1), control, nn_module)
    a2 = fit_alpha(colmatrix(train.data, c2), colmatrix(train[k], c2), a1.train, colmatrix(valid.data, c2), control, nn_module)
    a3 = fit_alpha(colmatrix(train.data, c3), colmatrix(train[j], c3), a2.train, colmatrix(valid.data, c3), control, nn_module)
    a4 = fit_alpha(colmatrix(train.data, c4), colmatrix(train[i], c4), a3.train, colmatrix(valid.data, c4), control, nn_module)
    Dict("alpha1" => a1.valid, "alpha2" => a2.valid, "alpha3" => a3.valid, "alpha4" => a4.valid)
end

function estimate_alphas(cd, folds, plist, fn, keyfn, nn_module, control)
    isempty(plist) && return nothing
    out = Dict{String, Dict{String, Vector{Float64}}}()
    n = nrow(cd.data)
    for (v, fold) in enumerate(folds)
        train = training(cd, folds, v)
        valid = validation(cd, folds, v)
        for p in plist
            key = keyfn(p)
            a = fn(train, valid, cd.vars, nn_module, p, control)
            slot = get!(out, key) do
                Dict(s => fill(NaN, n) for s in keys(a))
            end
            for (s, vals) in a
                slot[s][fold.validation_set] = vals
            end
        end
    end
    out
end

estimate_phi_n_alpha(cd, folds, params, nn_module, control) =
    estimate_alphas(cd, folds, params[:natural], phi_n_alpha, param_key, nn_module, control)
estimate_phi_r_alpha(cd, folds, params, nn_module, control) =
    estimate_alphas(cd, folds, params[:randomized], phi_r_alpha, param_key, nn_module, control)
