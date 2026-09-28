# Outcome-regression learners.
#
# R crumble fits every regression with mlr3superlearner; its default library is
# "glm", a main-effects GLM: logistic for a binary outcome, Gaussian for the
# continuous pseudo-outcomes of the later stages. We implement that learner and a
# "saturated" learner (cell means over the distinct covariate patterns), which is
# nonparametric when every regressor is discrete. A super learner over several
# libraries is not implemented, so exactly one learner is accepted.

abstract type LearnerFit end

struct GLMFit <: LearnerFit
    beta::Vector{Float64}
    logistic::Bool
end

struct SaturatedFit <: LearnerFit
    cells::Dict{Vector{Float64}, Float64}
    fallback::GLMFit
end

design(X::AbstractMatrix) = hcat(ones(size(X, 1)), X)

# Least squares that tolerates collinear columns (e.g. both dummies of a one-hot
# factor next to the intercept): minimum-norm solution via pivoted QR.
function ols_beta(D::AbstractMatrix, y::AbstractVector)
    F = qr(D, ColumnNorm())
    r = count(abs.(diag(F.R)) .> 1e-9 * abs(F.R[1, 1]))
    beta = zeros(size(D, 2))
    beta[F.p[1:r]] = UpperTriangular(F.R[1:r, 1:r]) \ (F.Q' * y)[1:r]
    beta
end

# Logistic regression by iteratively reweighted least squares. A small ridge term
# keeps the iteration finite under separation, which occurs easily in small cells.
function logistic_beta(D::AbstractMatrix, y::AbstractVector; ridge = 1e-8, maxit = 100, tol = 1e-10)
    p = size(D, 2)
    beta = zeros(p)
    for _ in 1:maxit
        eta = D * beta
        mu = @. 1 / (1 + exp(-eta))
        w = @. max(mu * (1 - mu), 1e-10)
        H = D' * (w .* D) + ridge * I
        step = H \ (D' * (y .- mu) .- ridge .* beta)
        beta .+= step
        maximum(abs, step) < tol && break
    end
    beta
end

function fit_learner(X::AbstractMatrix, y::AbstractVector, learner::String; binary::Bool)
    D = design(X)
    g = GLMFit(binary ? logistic_beta(D, y) : ols_beta(D, y), binary)
    learner == "glm" && return g
    if learner == "saturated"
        sums = Dict{Vector{Float64}, Tuple{Float64, Int}}()
        for i in axes(X, 1)
            key = Vector{Float64}(X[i, :])
            s, c = get(sums, key, (0.0, 0))
            sums[key] = (s + y[i], c + 1)
        end
        return SaturatedFit(Dict(k => s / c for (k, (s, c)) in sums), g)
    end
    throw(ArgumentError("Crumble.jl: unknown learner \"$learner\"; use \"glm\" or \"saturated\"."))
end

function predict_learner(f::GLMFit, X::AbstractMatrix)
    eta = design(X) * f.beta
    f.logistic ? @.(1 / (1 + exp(-eta))) : eta
end

# A pattern not seen in training (possible after shifting the treatment) falls
# back on the main-effects GLM rather than on an arbitrary value.
function predict_learner(f::SaturatedFit, X::AbstractMatrix)
    fb = predict_learner(f.fallback, X)
    [get(f.cells, Vector{Float64}(X[i, :]), fb[i]) for i in axes(X, 1)]
end
