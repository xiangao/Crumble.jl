# Sequential outcome regressions, following R crumble's theta().
#
# Natural functional (j, k, l):
#   mu      = E[Y | A, W, M, Z]            fitted on observed rows
#   b3      = mu evaluated at data_j
#   theta2  = E[b3 | A, W, Z],   b2 = theta2 at data_k
#   theta1  = E[b2 | A, W],      b1 = theta1 at data_l
# so that mean(b1) = E_W  sum_z p(z | l, W) sum_m p(m | k, W, z) mu(j, W, z, m).
#
# Randomized functional (i, j, k, l), with data_i carrying Z' (a permuted copy of Z):
#   b4 = mu at data_i;  theta3 = E[b4 | A, W, M], b3 at data_j;
#   theta2 = E[b3 | A, W, Z], b2 at data_k;  theta1 = E[b2 | A, W], b1 at data_l.
#
# Each stage also returns its fit at the observed validation data ("fit*"), which
# the influence function subtracts from the next stage's b.

function stage(train_df, pseudo, cols, learner)
    fit_learner(colmatrix(train_df, cols), pseudo, learner; binary = false)
end

pred(f, df, cols) = predict_learner(f, colmatrix(df, cols))

function theta(train, valid, vars::CrumbleVars, params, learner::String)
    Y = train.data[:, vars.Y]
    obs = vars.C === nothing ? trues(nrow(train.data)) : (train.data[:, vars.C] .== 1)
    obs = obs .& .!ismissing.(Y)
    cy = present(train.data, [vars.A; vars.W; vars.M; vars.Z])
    binary = is_binary(Y[obs])
    mu = fit_learner(colmatrix(train.data[obs, :], cy), Float64.(Y[obs]), learner; binary = binary)

    cz = present(train.data, [vars.A; vars.W; vars.Z])
    cm = present(train.data, [vars.A; vars.W; vars.M])
    ca = present(train.data, [vars.A; vars.W])

    out = Dict{String, Any}()

    if !isempty(params[:natural])
        vals = Dict{String, Dict{String, Vector{Float64}}}()
        for p in params[:natural]
            j, k, l = Symbol(p["j"]), Symbol(p["k"]), Symbol(p["l"])
            b3_train = pred(mu, train[j], cy)
            t2 = stage(train.data, b3_train, cz, learner)
            b2_train = pred(t2, train[k], cz)
            t1 = stage(train.data, b2_train, ca, learner)
            vals[param_key(p)] = Dict(
                "fit3" => pred(mu, valid.data, cy), "b3" => pred(mu, valid[j], cy),
                "fit2" => pred(t2, valid.data, cz), "b2" => pred(t2, valid[k], cz),
                "fit1" => pred(t1, valid.data, ca), "b1" => pred(t1, valid[l], ca),
            )
        end
        out["n"] = vals
    end

    if !isempty(params[:randomized])
        vals = Dict{String, Dict{String, Vector{Float64}}}()
        for p in params[:randomized]
            i, j, k, l = Symbol(p["i"]), Symbol(p["j"]), Symbol(p["k"]), Symbol(p["l"])
            b4_train = pred(mu, train[i], cy)
            t3 = stage(train.data, b4_train, cm, learner)
            b3_train = pred(t3, train[j], cm)
            t2 = stage(train.data, b3_train, cz, learner)
            b2_train = pred(t2, train[k], cz)
            t1 = stage(train.data, b2_train, ca, learner)
            vals[param_key(p)] = Dict(
                "fit4" => pred(mu, valid.data, cy), "b4" => pred(mu, valid[i], cy),
                "fit3" => pred(t3, valid.data, cm), "b3" => pred(t3, valid[j], cm),
                "fit2" => pred(t2, valid.data, cz), "b2" => pred(t2, valid[k], cz),
                "fit1" => pred(t1, valid.data, ca), "b1" => pred(t1, valid[l], ca),
            )
        end
        out["r"] = vals
    end
    out
end

# Cross-fit: every quantity is predicted on the fold it was not trained on, and
# the folds are stitched back into full-length vectors in the original row order.
function estimate_theta(cd::CrumbleData, folds, params, learner::String)
    n = nrow(cd.data)
    out = Dict{String, Dict{String, Dict{String, Vector{Float64}}}}()
    for (v, fold) in enumerate(folds)
        th = theta(training(cd, folds, v), validation(cd, folds, v), cd.vars, params, learner)
        for (family, vals) in th, (key, quantities) in vals
            fam = get!(out, family, Dict{String, Dict{String, Vector{Float64}}}())
            slot = get!(fam, key) do
                Dict(q => fill(NaN, n) for q in keys(quantities))
            end
            for (q, x) in quantities
                slot[q][fold.validation_set] = x
            end
        end
    end
    out
end
