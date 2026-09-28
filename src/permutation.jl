# Z' for the randomized functionals: each observation receives the Z of another
# observation with similar (A, W), so Z' has (approximately) the law of Z given
# (A, W) but is independent of the observation's own M. R crumble solves the
# assignment problem  min sum_ij D_ij P_ij  over permutation matrices with
# trace(P) = 0 (no unit may keep its own Z). The trace constraint matters: without
# it the identity has zero cost, Z' = Z, and the randomized effects collapse to
# natural ones. Here the same problem is solved by the Hungarian algorithm with
# the diagonal priced out.

# Hungarian algorithm (shortest augmenting path with potentials), O(n^3).
# Returns p with p[i] = column assigned to row i.
function hungarian(C::AbstractMatrix{Float64})
    n = size(C, 1)
    size(C, 2) == n || throw(ArgumentError("hungarian: cost matrix must be square"))
    # Index 1 of u/v/p/way plays the role of the dummy row/column 0.
    u = zeros(n + 1); v = zeros(n + 1)
    p = zeros(Int, n + 1); way = zeros(Int, n + 1)
    for i in 1:n
        p[1] = i
        j0 = 1
        minv = fill(Inf, n + 1)
        used = falses(n + 1)
        while true
            used[j0] = true
            i0 = p[j0]
            delta = Inf
            j1 = 0
            for j in 2:n+1
                used[j] && continue
                cur = C[i0, j - 1] - u[i0 + 1] - v[j]
                if cur < minv[j]
                    minv[j] = cur
                    way[j] = j0
                end
                if minv[j] < delta
                    delta = minv[j]
                    j1 = j
                end
            end
            for j in 1:n+1
                if used[j]
                    u[p[j] + 1] += delta
                    v[j] -= delta
                else
                    minv[j] -= delta
                end
            end
            j0 = j1
            p[j0] == 0 && break
        end
        while true
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            j0 == 1 && break
        end
    end
    assignment = zeros(Int, n)
    for j in 2:n+1
        assignment[p[j]] = j - 1
    end
    assignment
end

function derangement_by_distance(AW::AbstractMatrix{Float64})
    n = size(AW, 1)
    n < 2 && throw(ArgumentError("Crumble.jl: need at least two observations per Z' fold."))
    D = pairwise(Euclidean(), AW, dims = 1)
    mx = maximum(D)
    mx > 0 && (D ./= mx)
    for i in 1:n
        D[i, i] = 1e6
    end
    # Break ties among equally distant candidates at random, so that the match
    # does not depend on row order.
    D .+= 1e-9 .* rand(n, n)
    hungarian(D)
end

function set_zp(cd::CrumbleData, folds::Int)
    fold_obj = make_folds(nrow(cd.data), folds, cd.vars.id === nothing ? nothing : cd.data[:, cd.vars.id])
    AW = Matrix{Float64}(cd.data[:, present(cd.data, [cd.vars.A; cd.vars.W])])
    zp = DataFrame([z => Vector{Float64}(undef, nrow(cd.data)) for z in cd.vars.Z])
    for fold in fold_obj
        idx = fold.validation_set
        perm = derangement_by_distance(AW[idx, :])
        for z in cd.vars.Z
            zp[idx, z] = Float64.(cd.data[idx[perm], z])
        end
    end
    zp
end

function add_zp(cd::CrumbleData, moc, control)
    moc === nothing && return cd
    zp = set_zp(cd, control.zprime_folds)
    cd.data_0zp = copy(cd.data_0)
    cd.data_1zp = copy(cd.data_1)
    for z in cd.vars.Z
        cd.data_0zp[!, z] = zp[:, z]
        cd.data_1zp[!, z] = zp[:, z]
    end
    cd
end
