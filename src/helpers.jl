function one_hot_encode(data::DataFrame, cols::Vector{Symbol})
    result = DataFrame()
    for col in cols
        vals = sort(unique(skipmissing(data[!, col])))
        for v in vals
            colname = Symbol("$(col)_$(v)")
            result[!, colname] = Int.(data[!, col] .== v)
        end
    end
    return result
end

present(df::DataFrame, cols) = [c for c in cols if hasproperty(df, c)]
cvec(C) = C === nothing ? Symbol[] : [C]

# "data_1zp","data_1","data_0","data_0" -> "1100"; "data_1","data_0","data_0" -> "100".
param_key(p) = join(replace(p[k], "data_" => "", "zp" => "") for k in ("i", "j", "k", "l") if haskey(p, k))

cluster_ids(cd) = cd.vars.id === nothing ? collect(1:nrow(cd.data)) :
    let ids = cd.data[:, cd.vars.id], u = Dict(v => i for (i, v) in enumerate(unique(ids)))
        [u[v] for v in ids]
    end

# V-fold split. With cluster ids, whole clusters go to one fold. With a binary
# strata vector (the outcome, as in R crumble), each stratum is spread evenly.
function make_folds(n::Int, V::Int, id = nothing, strata = nothing)
    V == 1 && return [CrossFitFold(collect(1:n), collect(1:n))]
    units = id === nothing ? [[i] for i in 1:n] : collect(values(group_rows(id)))
    labels = strata === nothing ? fill(0, length(units)) :
        [strata[u[1]] === missing ? -1 : Int(round(Float64(strata[u[1]]))) for u in units]
    assign = zeros(Int, length(units))
    for s in unique(labels)
        idx = shuffle(findall(==(s), labels))
        for (r, u) in enumerate(idx)
            assign[u] = mod1(r, V)
        end
    end
    folds = CrossFitFold[]
    for v in 1:V
        valid = sort(reduce(vcat, units[assign .== v]; init = Int[]))
        push!(folds, CrossFitFold(setdiff(1:n, valid), valid))
    end
    folds
end

function group_rows(id)
    g = Dict{Any, Vector{Int}}()
    for (i, v) in enumerate(id)
        push!(get!(g, v, Int[]), i)
    end
    g
end

function is_normalized(x; tolerance=sqrt(eps()))
    return abs(mean(x) - 1) < tolerance
end

function normalize_weights(x)
    if is_normalized(x)
        return x
    end
    return x ./ mean(x)
end

function is_binary(x)
    unique_vals = unique(skipmissing(x))
    return length(unique_vals) <= 2 && all(v -> v in (0, 1, 0.0, 1.0), unique_vals)
end
