struct CrumbleDataset
    data::Dict{String, Matrix{Float64}}
    n::Int
end

function CrumbleDataset(data_dict::NamedTuple, vars::Vector{Symbol}, device::String)
    result = Dict{String, Matrix{Float64}}()
    for df_name in ("data", "data_0", "data_1", "data_0zp", "data_1zp")
        if hasproperty(data_dict, Symbol(df_name))
            df = getproperty(data_dict, Symbol(df_name))
            if ncol(df) > 0 && nrow(df) > 0
                available = [v for v in vars if hasproperty(df, v)]
                if !isempty(available)
                    result[df_name] = Matrix{Float64}(df[:, available])
                end
            end
        end
    end
    n = haskey(result, "data") ? size(result["data"], 1) : 0
    return CrumbleDataset(result, n)
end

function Base.length(ds::CrumbleDataset)
    return ds.n
end

function get_batch(ds::CrumbleDataset, indices::AbstractVector{Int})
    return Dict(k => v[indices, :] for (k, v) in ds.data)
end
