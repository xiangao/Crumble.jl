function train_alpha(train_data, valid_data, vars::Vector{Symbol}, architecture, f_loss, weights, control::CrumbleControl)
    ds = CrumbleDataset(train_data, vars, control.device)
    n = length(ds)
    n == 0 && return (train=zeros(n), valid=zeros(n))

    d_in = haskey(ds.data, "data") ? size(ds.data["data"], 2) : 0
    d_in == 0 && return (train=zeros(n), valid=zeros(n))

    model = architecture(d_in)

    w = weights !== nothing ? weights : 1.0

    opt_state = Flux.setup(Adam(0.01), model)

    batches = [ds.data["data"][i:min(i+control.batch_size-1, n), :] for i in 1:control.batch_size:n]

    for epoch in 1:control.epochs
        lr = control.learning_rate * min(1.0, epoch / 10.0)
        Flux.adjust!(opt_state, lr)

        for batch_data in batches
            batch_idx = 1:size(batch_data, 1)
            x = batch_data'
            loss_val = mean(model(x).^2 .- 2.0 .* w .* f_loss(model, x))

            grads = Flux.gradient(model) do m
                mean(m(x).^2 .- 2.0 .* w .* f_loss(m, x))
            end
            Flux.update!(opt_state, model, grads[1])
        end
    end

    Flux.testmode!(model)

    train_x = haskey(ds.data, "data") ? ds.data["data"]' : zeros(d_in, 0)
    train_preds = model(train_x)[:, 1]

    valid_x = haskey(valid_data, :data) ? Matrix{Float64}(valid_data.data[:, vars])' : zeros(d_in, 0)
    valid_preds = model(valid_x)[:, 1]

    return (train=train_preds, valid=valid_preds)
end

function Alpha(train, valid, vars::Vector{Symbol}, architecture, f_loss, weights, control::CrumbleControl)
    result = train_alpha(train, valid, vars, architecture, f_loss, weights, control)
    return result
end
