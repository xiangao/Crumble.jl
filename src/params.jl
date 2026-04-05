const natural = Dict(
    :natural => [
        Dict("j" => "data_1", "k" => "data_1", "l" => "data_1"),
        Dict("j" => "data_1", "k" => "data_0", "l" => "data_0"),
        Dict("j" => "data_0", "k" => "data_0", "l" => "data_0"),
    ],
    :randomized => Dict{String, Any}[],
)

const organic = Dict(
    :natural => [
        Dict("j" => "data_1", "k" => "data_0", "l" => "data_1"),
        Dict("j" => "data_0", "k" => "data_0", "l" => "data_0"),
        Dict("j" => "data_1", "k" => "data_1", "l" => "data_1"),
    ],
    :randomized => Dict{String, Any}[],
)

const randomized = Dict(
    :natural => Dict{String, Any}[],
    :randomized => [
        Dict("i" => "data_1zp", "j" => "data_1", "k" => "data_0", "l" => "data_0"),
        Dict("i" => "data_0zp", "j" => "data_0", "k" => "data_0", "l" => "data_0"),
        Dict("i" => "data_1zp", "j" => "data_1", "k" => "data_1", "l" => "data_1"),
    ],
)

const recanting_twin = Dict(
    :natural => [
        Dict("j" => "data_1", "k" => "data_1", "l" => "data_1"),
        Dict("j" => "data_0", "k" => "data_1", "l" => "data_1"),
        Dict("j" => "data_0", "k" => "data_1", "l" => "data_0"),
        Dict("j" => "data_0", "k" => "data_0", "l" => "data_0"),
    ],
    :randomized => [
        Dict("i" => "data_0zp", "j" => "data_0", "k" => "data_1", "l" => "data_1"),
        Dict("i" => "data_0zp", "j" => "data_0", "k" => "data_1", "l" => "data_0"),
        Dict("i" => "data_0zp", "j" => "data_1", "k" => "data_1", "l" => "data_1"),
    ],
)
