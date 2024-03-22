function build_testing_dataloaders(data_path, seq_len, seed, dataset_type, verbose; forecast_horizon = nothing)
    verbose && @info "Generating dataloaders"

    file_path = data_path
    verbose && @info "Using data from: $file_path"
    rng = Random.MersenneTwister(seed)

    if isnothing(forecast_horizon)
        d_test = dataset_type(file_path, rng, seq_len)
    else
        d_test = dataset_type(file_path, rng, seq_len, forecast_horizon)
    end
    dataloader_test = MLUtils.DataLoader(d_test, batchsize=numobs(d_test), buffer=false, parallel=false,
                                            partial=false, rng=rng, shuffle=false)

    return dataloader_test
end

function load_and_run_model(model, x, args, forecast_horizon = 0)
    @unpack model_type = args
    load_and_run_model(model, x, args, forecast_horizon, model_type)
end

function load_and_run_model(model, x, args, forecast_horizon, model_type::GOKU)
    # load model parameters
    @unpack best_ps = args
    load_model_weights!(model, best_ps)

    @unpack dt, seq_len, win_len, continuity_term, multiple_shooting = args
    training_mode = if multiple_shooting
        MultipleShooting(win_len, seq_len, continuity_term)
    else 
        SingleShooting(seq_len)
    end

    t = range(0.f0, step = dt, length = training_mode.win_len + forecast_horizon)
    x̂, ẑ, θ̂ = run_model(model, x, t, training_mode)
end

function load_and_run_model(model, x, args, forecast_horizon, model_type::Union{LatentODE, LSTMModel})
    # load model parameters
    @unpack best_ps, dt, seq_len = args
    load_model_weights!(model, best_ps)

    training_mode = SingleShooting(seq_len)
    t = range(0.f0, step = dt, length = training_mode.win_len + forecast_horizon)
    x̂, ẑ, θ̂ = run_model(model, x, t, training_mode)
end

function run_model(model, x, t, training_mode::MultipleShooting)
    X̂, μ, logσ² = model(x, t, training_mode)
    x̂, ẑ, l̂ = X̂
    ẑ₀, θ̂ = l̂
    # Join all windows from all outputs x̂
    batch_size = size(x, 2)
    win_len = training_mode.win_len
    win_per_sample = Int(size(x̂, 2)/batch_size)

    # indexes of windows for sample j excluding the last window of each sample
    inds(j) = (1:(Int(win_per_sample) - 1)) .+ (j-1)*win_per_sample
    x̂_joint = [hcat([x̂[:, i, 1:(win_len-1)] for i in inds(j)]..., x̂[:, Int(j*win_per_sample), :]) for j in 1:batch_size]
    x̂_joint = Flux.stack(x̂_joint, dims=2)
    x̂ = x̂_joint

    ẑ_joint = [hcat([ẑ[:, i, 1:(win_len-1)] for i in inds(j)]..., ẑ[:, Int(j*win_per_sample), :]) for j in 1:batch_size]
    ẑ_joint = Flux.stack(ẑ_joint, dims=2)
    ẑ = ẑ_joint

    return x̂, ẑ, θ̂
end

function run_model(model, x, t, training_mode::SingleShooting)
    X̂, μ, logσ² = model(x, t, training_mode)
    x̂, ẑ, l̂ = X̂
    ẑ₀, θ̂ = l̂
    return x̂, ẑ, θ̂
end

function model_name(model_type::GOKU, multiple_shooting)
    base_model = model_type == GOKU_attention() ? "Attention" : "Basic"
    shooting = multiple_shooting ? "Multiple Shooting" : "Single Shooting"
    return "GOKU $base_model with $shooting"
end

NRMSE(x, x̂) = vec(sqrt.(mean((x .- x̂).^2, dims = (1, 3)))./mean(abs.(x)))

function get_scores(df0, testing_data, forecast_horizon)

    # get rid of possible duplicates
    @rtransform!(df0, :val_rec_losses = :losses_hist.val_rec_losses)
    unique!(df0, :val_rec_losses)

    @info "$(nrow(df0)) files to process"
    @unpack seq_len = df0[1,:]
    df0.verbose .= false
    seed = 3
    dataloader = build_testing_dataloaders(testing_data, seq_len, seed, TestingDataset_forecast,
                                            true, forecast_horizon = forecast_horizon)
    (x, z, θ), (x_future, z_future) = first(dataloader)

    df = DataFrame(score_rec = Array{Float32}[], score_for = Array{Float32}[],
                    model = String[], training_samples = Int64[], seed = Int64[], file = String[])

    progress =  Progress(nrow(df0))
    for (i, row) in enumerate(eachrow(df0))
        progress_values = [(:Processing,  row.path), (:File, "$i/$(nrow(df0))")]
        next!(progress; showvalues=progress_values)
        @unpack seq_len, training_samples, seed = row
        m_name = model_name(row)
                            
        model = build_model!(row, forecast_horizon);
        Random.seed!(3)
        x̂, ẑ, θ̂ = load_and_run_model(model, x, row, forecast_horizon)
        x̂, x̂_future = x̂[:, :, 1:seq_len], x̂[:, :, seq_len+1:end]

        score_rec = NRMSE(x, x̂)
        score_for = NRMSE(x_future, x̂_future)

        push!(df, (score_rec, score_for, m_name,
                    training_samples, seed, row.path))
    end
    return df
end

function model_name(row)        
    @unpack model_type = row
    if model_type isa GOKU
        @unpack multiple_shooting = row
    else
        multiple_shooting = nothing
    end
    model_name(model_type, multiple_shooting)
end