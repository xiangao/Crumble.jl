function shift_data(data::DataFrame, trt::Vector{Symbol}, cens::Union{Symbol, Nothing}, shift::Union{Function, Nothing})
    data = shift_cens(data, cens)
    if shift === nothing
        return data
    end
    if length(trt) > 1
        return shift_trt_multivariate(data, trt, shift)
    end
    return shift_trt_single(data, trt[1], shift)
end

function shift_cens(data::DataFrame, cens::Union{Symbol, Nothing})
    if cens === nothing
        return data
    end
    out = copy(data)
    out[:, cens] .= 1
    return out
end

function shift_trt_single(data::DataFrame, trt::Symbol, f::Function)
    out = copy(data)
    out[:, trt] = f(data, trt)
    return out
end

function shift_trt_multivariate(data::DataFrame, trt::Vector{Symbol}, f::Function)
    out = copy(data)
    new_vals = f(data, trt)
    for (i, a) in enumerate(trt)
        out[:, a] = new_vals[:, i]
    end
    return out
end
