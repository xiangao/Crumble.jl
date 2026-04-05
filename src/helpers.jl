function recombine_theta(x, folds)
    result = Dict{String, Any}()
    
    # Process natural effects (n)
    if haskey(x[1], "n") && x[1]["n"] !== nothing
        bs_dict = Dict{String, Any}()
        natural_dict = Dict{String, Any}()
        
        for fold_result in x
            for (key, val) in fold_result["n"]
                if !haskey(bs_dict, key)
                    bs_dict[key] = Dict{String, Vector{Float64}}()
                    natural_dict[key] = Dict{String, Vector{Float64}}()
                end
                for (subkey, subval) in val
                    if startswith(subkey, "b")
                        if !haskey(bs_dict[key], subkey)
                            bs_dict[key][subkey] = Float64[]
                        end
                        append!(bs_dict[key][subkey], subval)
                    else
                        if !haskey(natural_dict[key], subkey)
                            natural_dict[key][subkey] = Float64[]
                        end
                        append!(natural_dict[key][subkey], subval)
                    end
                end
            end
        end
        
        result["theta_n"] = (bs=bs_dict, natural=natural_dict, weights=Dict())
    else
        result["theta_n"] = (bs=Dict(), natural=Dict(), weights=Dict())
    end
    
    # Process randomized effects (r)
    if haskey(x[1], "r") && x[1]["r"] !== nothing
        bs_dict = Dict{String, Any}()
        
        for fold_result in x
            for (key, val) in fold_result["r"]
                if !haskey(bs_dict, key)
                    bs_dict[key] = Dict{String, Vector{Float64}}()
                end
                for (subkey, subval) in val
                    if startswith(subkey, "b")
                        if !haskey(bs_dict[key], subkey)
                            bs_dict[key][subkey] = Float64[]
                        end
                        append!(bs_dict[key][subkey], subval)
                    end
                end
            end
        end
        
        result["theta_r"] = (bs=bs_dict, natural=Dict(), weights=Dict())
    else
        result["theta_r"] = (bs=Dict(), natural=Dict(), weights=Dict())
    end
    
    return (theta_n=get(result, "theta_n", (bs=Dict(), natural=Dict(), weights=Dict())), 
            theta_r=get(result, "theta_r", (bs=Dict(), natural=Dict(), weights=Dict())))
end

function recombine_alpha(x, folds)
    result = Dict{String, Dict{String, Vector{Float64}}}()
    
    for fold_result in x
        for (key, val) in fold_result
            if !haskey(result, key)
                result[key] = Dict(
                    "alpha1" => Float64[],
                    "alpha2" => Float64[],
                    "alpha3" => Float64[],
                )
            end
            for (alpha_key, alpha_val) in val
                if alpha_key != "jkl" && alpha_key != "ijkl"
                    append!(result[key][alpha_key], alpha_val)
                end
            end
        end
    end
    
    return result
end

function make_folds(n::Int, V::Int, id::Union{Vector{<:Any}, Nothing}=nothing, strata::Union{Vector{<:Any}, Nothing}=nothing)
    folds = CrossFitFold[]
    if V == 1
        push!(folds, CrossFitFold(collect(1:n), collect(1:n)))
        return folds
    end

    shuffled = shuffle(1:n)
    fold_sizes = [n ÷ V + (i <= n % V ? 1 : 0) for i in 1:V]
    starts = cumsum([1; fold_sizes[1:end-1]])
    for v in 1:V
        val_idx = shuffled[starts[v]:starts[v]+fold_sizes[v]-1]
        train_idx = setdiff(1:n, val_idx)
        push!(folds, CrossFitFold(train_idx, val_idx))
    end
    return folds
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
