function build_model!(config, forecast_horizon, model_type::LatentODE)
    @unpack model_type, cuda, verbose, resume, seed = config
    seed > 0 && Random.seed!(seed)

    # Get model hyperparameters
    @unpack input_dim, hidden_dim_resnet, rnn_input_dim, rnn_output_dim, z_dim, augment_dim = config
    @unpack latent_dim_z₀, latent_to_diffeq_dim, z₀_activation, general_activation, output_activation, init = config
    @unpack dt, seq_len, node_hidden_dim = config

    model_args = @dict(hidden_dim_resnet, rnn_input_dim, rnn_output_dim,
                        latent_dim_z₀, latent_to_diffeq_dim, z_dim, augment_dim,
                        z₀_activation, general_activation,
                        output_activation, init)

    # Set up device
    if cuda && has_cuda_gpu()
        device = gpu
        verbose && @info "Computations on GPU"
    else
        device = cpu
        verbose && @info "Computations on CPU"
    end

    # Create model
    verbose && @info "Building model"
    f = Chain(Dense(z_dim + augment_dim, node_hidden_dim, relu),
                    Dense(node_hidden_dim, node_hidden_dim, relu),
                    Dense(node_hidden_dim, z_dim + augment_dim)) |> device

    t = range(0.f0, step = Float32(dt), length = seq_len + forecast_horizon)

    tspan = (t[1], t[end])
    node = NeuralODE(f, tspan, alg=Tsit5(), saveat = t)
    diffeq = AugmentedNDELayer(node, augment_dim)

    encoder_layers, decoder_layers = default_layers(model_type, input_dim, diffeq; device=device, model_args...)
    model = LatentDiffEqModel(model_type, encoder_layers, decoder_layers)

    # Load model parameters if resuming an unfinished training
    if resume && isfile("best_ps_for_resuming.jld2")
        best_ps = load("best_ps_for_resuming.jld2", "best_ps")
        load_model_weights!(model, best_ps)
        verbose && println("Moedl loaded")
    end

    !haskey(config, "device") && merge!(config, @dict device)

    return model
end

function init_training_state(model, config, model_type::LatentODE)
    @unpack dt, seq_len = config
    @unpack patience_lr = config

    training_mode = SingleShooting(seq_len)
    t = range(0.f0, step = dt, length = seq_len)
    t0 = time_ns()

    training_state = TrainingState2(
        best_ps = extract_model_weights(model),
        best_ps_train = extract_model_weights(model),
        t = t,
        training_mode = training_mode,
        t0 = t0,
        patience_lr = patience_lr,
    )

    return training_state
end

@with_kw mutable struct TrainingState2
    best_ps::NTuple{6, Params{Zygote.Buffer{Any, Vector{Any}}}}
    best_ps_train::NTuple{6, Params{Zygote.Buffer{Any, Vector{Any}}}}
    train_losses::Vector{Float32} = Float32[]
    val_rec_losses::Vector{Float32} = Float32[]
    best_loss_val::Float32 = Inf32
    best_loss::Float32 = Inf32
    best_loss_val_plateau::Float32 = Inf32
    loss::Float32 = 0f0
    early_stop::Bool = false
    early_stopping::Int = 0
    patience_lr::Int = 50
    plateau_counter::Int = 0
    lr_counter::Int = 1
    sample_counter::Int = 0
    batch_counter::Int = 0
    epoch::Int = 0
    lr::Float32 = 0f0
    t::StepRangeLen{Float32, Float64, Float64, Int64}
    training_mode::Union{MultipleShooting, SingleShooting}
    t0::UInt64
end

function save_data!(best_ps, best_ps_train, losses_hist, config, model_type::LatentODE)
    merge!(config, @dict best_ps best_ps_train losses_hist)
    config = tostringdict(config)
    @unpack experiments_name, seed = config
    @unpack model_type, z_dim, training_samples = config

    prefix = "$(now())"
    modeltype = model_type |> typeof |> string
    sname = savename(prefix, (@dict modeltype training_samples seed), "jld2")
    tagsave(datadir("exp_pro", experiments_name, sname), config)
    @info "Output saved: $sname"
    return datadir("exp_pro", experiments_name, sname)
end