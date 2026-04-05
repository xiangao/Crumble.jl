function assert_not_missing(data::DataFrame, A, W, M, Z, C)
    cols = Symbol.([A..., W..., M..., Z..., C...])
    cols = [c for c in cols if c !== nothing]
    for col in cols
        if hasproperty(data, col) && any(ismissing.(data[:, col]))
            throw(ArgumentError("Missing data found in treatment/covariate/mediator/observed nodes"))
        end
    end
    return true
end

function assert_binary_0_1(data::DataFrame, x::Union{Symbol, Nothing})
    if x === nothing || !hasproperty(data, x)
        return true
    end
    var = data[:, x]
    unique_vals = unique(skipmissing(var))
    if length(unique_vals) == 2 && !all(v -> v in (0, 1), unique_vals)
        throw(ArgumentError("The outcome contains exactly two unique values, but they are not 0 and 1"))
    end
    return true
end

function assert_effect_type(moc, effect::String)
    if moc === nothing && effect == "RT"
        throw(ArgumentError("Must provide mediator-outcome confounders for recanting twins"))
    end
    if moc === nothing && effect == "RI"
        throw(ArgumentError("Must provide mediator-outcome confounders for interventional effects"))
    end
    if moc !== nothing && effect in ("N", "O", "D")
        throw(ArgumentError("Must not provide mediator-outcome confounders for natural, organic, or decision theoretic effects"))
    end
    return true
end
