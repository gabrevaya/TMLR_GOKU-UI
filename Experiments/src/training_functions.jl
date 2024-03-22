function training_pipeline(config, wandb)
    lg = WandbLogger(
        project = "$(config[:experiments_name])",
        name = "$(config[:name])-$(now())",
        config = config,
    )

    with_logger(lg) do

        # Define metrics
        Wandb.wandb.define_metric("Training/Loss", summary="min", goal="minimize")
        Wandb.wandb.define_metric("Testing/RecValLoss", summary="min", goal="minimize")
        Wandb.wandb.define_metric("Testing/ParamValLoss", summary="min", goal="minimize")

        try
            # Build the model, data, optimization problem and schedules
            model, data_loaders, opt_and_schedules = make!(config)

            # Train the model
            train!(model, data_loaders, opt_and_schedules, config, lg)

        finally
            Wandb.wandb.finish()
        end
    end
    close(lg)
end

function training_pipeline(config)
    # Build the model, data, optimization problem and schedules
    model, data_loaders, opt_and_schedules = make!(config)

    # Train the model
    train!(model, data_loaders, opt_and_schedules, config)
end

function make!(config)
    # Get dataloaders
    data_loaders = build_dataloaders!(config)

    # Build model
    model = build_model!(config)

    # Define optimizer and annealing schedule
    opt_and_schedules = define_optimizer_and_schedules(config)

    return model, data_loaders, opt_and_schedules
end

function build_dataloaders!(config)
    @unpack seed, batch_size, seq_len = config
    @unpack training_samples, verbose = config
    verbose && @info "Generating dataloaders"

    if haskey(config, "win_len")
        @unpack win_len = config
        seq_len = adjust_seq_len_for_multiple_shooting(seq_len, win_len)
    end
    
    file_path = get_file_path(config)
    verbose && @info "Using data from: $file_path"
    rng = Random.MersenneTwister(seed)

    # Check that the available data samples are enough
    data_size = h5open(file_path, "r") do file
        obj = file["training/high_dim_data"]
        size(obj)
    end
    input_dim, full_seq_len, full_samples = data_size
    @assert full_samples ≥ training_samples "Not enough samples in the dataset for the requested `training_samples`."

    d_train = TrainingDataset(file_path, rng, seq_len, training_samples)
    dataloader_train = MLUtils.DataLoader(d_train, batchsize=batch_size, buffer=false, parallel=false,
                                            partial=false, rng=rng, shuffle=true)

    d_val = ValidationDataset_HighDimOnly(file_path, rng, seq_len)  
    dataloader_val = MLUtils.DataLoader(d_val, batchsize=numobs(d_val), buffer=false, parallel=false,
                                            partial=false, rng=rng, shuffle=false)

    merge!(config, @dict input_dim full_seq_len full_samples seq_len)

    return dataloader_train, dataloader_val
end

function build_model!(config, forecast_horizon = 0)
    @unpack model_type = config
    build_model!(config, forecast_horizon, model_type)
end

function build_model!(config, forecast_horizon, model_type::GOKU)
    @unpack input_dim, diffeq, diffeq_args, cuda = config
    @unpack model_type, verbose, resume, seed = config
    seed > 0 && Random.seed!(seed)

    verbose && @info "Building differential equation"
    diffeq = diffeq(;diffeq_args...)
    diffeq_type, diffeq_dim, diffeq_state_size, diffeq_p_size = describe(diffeq)

    # Get model hyperparameters
    @unpack hidden_dim_resnet, rnn_input_dim, rnn_output_dim, latent_dim_z₀ = config
    @unpack latent_dim_θ, latent_to_diffeq_dim, general_activation, z₀_activation = config
    @unpack θ_activation, output_activation, init = config

    model_args = @dict(hidden_dim_resnet, rnn_input_dim, rnn_output_dim, latent_dim_z₀,
                        latent_dim_θ, latent_to_diffeq_dim, general_activation, z₀_activation,
                        θ_activation, output_activation, init)

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
    encoder_layers, decoder_layers = default_layers(model_type, input_dim, diffeq; device=device, model_args...)
    model = LatentDiffEqModel(model_type, encoder_layers, decoder_layers)

    # Load model parameters if resuming an unfinished training
    if resume && isfile("best_ps_for_resuming.jld2")
        best_ps = load("best_ps_for_resuming.jld2", "best_ps")
        load_model_weights!(model, best_ps)
        verbose && println("Moedl loaded")
    end

    !haskey(config, "diffeq_dim") && merge!(config, @dict diffeq_type diffeq_dim diffeq_state_size diffeq_p_size device)

    return model
end

function define_optimizer_and_schedules(config)
    ## Define optimizer
    @unpack optimizer, lr, decay_of_momentums, ϵ, weight_decay, verbose = config
    verbose && @info "Defining optimizers and schedules"

    opt = optimizer(lr, decay_of_momentums, ifelse(optimizer == ADAMW, weight_decay, ϵ))

    # Setup lr schedules
    @unpack lr_scheduler, warmup, lr, min_lr, period, γ = config
    start_lr = 1e-7 # don't actually start with lr = 0
    lr_warmup = WarmupLinear(start_lr, lr, warmup, warmup)
    if lr_scheduler == Cos4Exp
        lr_schedule = lr_scheduler(lr, min_lr, period, γ)
    elseif lr_scheduler == Exp
        lr_schedule = lr_scheduler(lr , γ)
    end

    # Setup KL annealing schedule
    # cyclic schedule
    # @unpack start_β, end_β, warmup_β, len_constant_β, period_β = config
    # s_linear = WarmupLinear(start_β, end_β, warmup_β, len_constant_β)
    # annealing_schedule = Loop(s_linear, period_β)
    
    # Exponential schedule
    @unpack initial_wait_β, annealing_γ, annealing_length = config
    annealing_schedule = Sequence(  1 => initial_wait_β,
                                    Exp(λ = 1.0, γ = annealing_γ) => annealing_length)

    return opt, lr_warmup, lr_schedule, annealing_schedule
end

function WarmupLinear(initial, final, warmup, length_constant=warmup)
    Sequence(Triangle(λ0 = initial, λ1 = final, period = 2 * warmup) => warmup,
            final => length_constant)
end

function train!(model, data_loaders, opt_and_schedules, config, lg=nothing)
    dataloader_train, dataloader_val = data_loaders
    @unpack verbose, epochs = config
    @unpack logging_and_scheduling_period = config
    verbose && @info "Starting training"
    
    # Initialize training state
    training_state = init_training_state(model, config)
    verbose && (progress = Progress(round(Int, epochs * numobs(dataloader_train) / logging_and_scheduling_period)))

    # Main training loop
    for epoch = 1:epochs
        training_state.early_stop && (early_stop_info(training_state); break) 

        for x in dataloader_train
            update_training_state!(training_state, model, x, opt_and_schedules, config, epoch)

            # Logging, lr scheduling and early stopping
            if (training_state.batch_counter % logging_and_scheduling_period == 0)
                update_validation_and_logging!(training_state, model, dataloader_val, opt_and_schedules, config, lg, progress)
                training_state.t0 = time_ns()
            end
        end
    end

    # Save data and clean up
    finalize_training!(training_state, progress, model, config)

    return nothing
end

function init_training_state(model, config)
    @unpack model_type = config
    init_training_state(model, config, model_type)
end

function init_training_state(model, config, model_type::GOKU)
    @unpack dt, win_len, seq_len, continuity_term = config
    @unpack multiple_shooting, patience_lr = config

    training_mode = if multiple_shooting
        MultipleShooting(win_len, seq_len, continuity_term)
    else 
        SingleShooting(seq_len)
    end

    t = range(0.f0, step = dt, length = training_mode.win_len)
    t0 = time_ns()

    training_state = TrainingState(
        best_ps = extract_model_weights(model),
        best_ps_train = extract_model_weights(model),
        t = t,
        training_mode = training_mode,
        t0 = t0,
        patience_lr = patience_lr,
    )

    return training_state
end

@with_kw mutable struct TrainingState
    best_ps::NTuple{5, Params{Zygote.Buffer{Any, Vector{Any}}}}
    best_ps_train::NTuple{5, Params{Zygote.Buffer{Any, Vector{Any}}}}
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

function update_training_state!(training_state, model, x, opt_and_schedules, config, epoch)
    opt, lr_warmup, lr_schedule, annealing_schedule = opt_and_schedules
    @unpack device, batch_size, warmup, variational = config

    # Set annealing factor
    β = 1 - annealing_schedule(training_state.batch_counter)

    # Set learning rate if in warmup phase
    if training_state.batch_counter ≤ warmup + 1
        lr = lr_warmup(training_state.batch_counter)
        # Flux.adjust!(opt, lr)
        opt[1].eta = lr
        training_state.lr = lr
    end

    loss = training_step!(model, x |> device, training_state.t, β, variational, training_state.training_mode, opt)

    training_state.epoch = epoch
    training_state.loss = loss
    training_state.sample_counter += batch_size
    training_state.batch_counter += 1
    opt_and_schedules = (opt, lr_warmup, lr_schedule, annealing_schedule)
end

function update_validation_and_logging!(training_state, model, dataloader_val, opt_and_schedules, config, lg, progress)
    opt, lr_warmup, lr_schedule, annealing_schedule = opt_and_schedules
    @unpack warmup, device = config

    # Evaluate model on validation set
    x_val, z_val, θ_val = first(dataloader_val) |> device
    val_rec_loss = evaluate_model(model, x_val, training_state.t, training_state.training_mode)

    # Update training state with losses and model parameters
    update_best_losses_and_ps!(training_state, model, val_rec_loss)

    # Log progress and visualize
    log_progress_and_visualize!(training_state, config, val_rec_loss, lg, progress)

    # Learning rate scheduling and early stopping
    if training_state.batch_counter > warmup
        update_lr_and_early_stopping!(training_state, val_rec_loss, config, opt, lr_schedule, progress)
    end
end

function update_best_losses_and_ps!(training_state, model, val_rec_loss)
    push!(training_state.train_losses, training_state.loss)
    push!(training_state.val_rec_losses, val_rec_loss)

    if val_rec_loss < training_state.best_loss_val
        training_state.best_loss_val = val_rec_loss
        training_state.best_ps = extract_model_weights(model)
    end

    if training_state.loss < training_state.best_loss
        training_state.best_loss = training_state.loss
        training_state.best_ps_train = extract_model_weights(model)
    end
end

function log_progress_and_visualize!(training_state, config, val_rec_loss, lg, progress)
    @unpack verbose = config
    time_per_period = elapsed_time(training_state.t0)

    if verbose
        progress_values = [(:epoch, training_state.epoch), (:batches, training_state.batch_counter), 
                           (:samples, training_state.sample_counter), (:loss, training_state.loss), 
                           (:val_rec_loss, val_rec_loss), (:time_per_period, time_per_period)]
        next!(progress; showvalues=progress_values)
    end

    # Log in WandB if provided
    if lg ≠ nothing
        log_in_wandb(lg, training_state.loss, val_rec_loss, training_state.best_loss_val, time_per_period,
                    training_state.lr, training_state.epoch, training_state.batch_counter, training_state.sample_counter)
    end
end

function update_lr_and_early_stopping!(training_state, val_rec_loss, config, opt, lr_schedule, progress)
    @unpack patience, patience_lr2, threshold, verbose = config
    if val_rec_loss < training_state.best_loss_val_plateau * (1 - threshold)
        training_state.plateau_counter = 0
        training_state.early_stopping = 0
        training_state.best_loss_val_plateau = val_rec_loss
    else
        training_state.plateau_counter += 1
        training_state.early_stopping += 1

        if training_state.plateau_counter > training_state.patience_lr
            training_state.lr_counter += 1
            training_state.patience_lr = patience_lr2
            lr = lr_schedule(training_state.lr_counter)
            # Flux.adjust!(opt, lr)
            opt[1].eta = lr
            training_state.lr = lr
            verbose && println("New lr: $lr")
            training_state.plateau_counter = 0
        end
    end

    # Early stopping check
    if training_state.early_stopping > patience
        @info "Early stopped after $(training_state.batch_counter) batches of training
               (during epoch $(training_state.epoch) | Elapsed time: $(show_time(progress))"
        training_state.early_stop = true
    end
end

function finalize_training!(training_state, progress, model, config)
    train_losses = training_state.train_losses
    val_rec_losses = training_state.val_rec_losses
    best_loss_val = training_state.best_loss_val
    best_loss = training_state.best_loss
    @unpack verbose, save_output = config

    # Save data
    if save_output
        losses_hist = @ntuple train_losses val_rec_losses best_loss_val best_loss
        fname = save_data!(training_state.best_ps, training_state.best_ps_train, losses_hist, config)
    end

    verbose && finishing_info(training_state, progress)

    # Delete checkpoint
    checkpoint_file = "best_ps_for_resuming.jld2"
    if isfile(checkpoint_file)
        rm(checkpoint_file)
    end
end

function save_data!(best_ps, best_ps_train, losses_hist, config)
    @unpack model_type = config
    save_data!(best_ps, best_ps_train, losses_hist, config, model_type)
end

function save_data!(best_ps, best_ps_train, losses_hist, config, model_type::GOKU)
    merge!(config, @dict best_ps best_ps_train losses_hist)
    config = tostringdict(config)
    @unpack experiments_name, diffeq_type, diffeq_dim, seed = config
    @unpack multiple_shooting, model_type, training_samples = config

    prefix = "$(now())_$(diffeq_type)_$(round(Int, diffeq_dim/2))"
    modeltype = model_type |> typeof |> string
    sname = savename(prefix, (@dict multiple_shooting modeltype training_samples seed), "jld2")
    tagsave(datadir("exp_pro", experiments_name, sname), config)
    @info "Output saved: $sname"
    return datadir("exp_pro", experiments_name, sname)
end

function log_in_wandb(lg, loss, val_rec_loss, best_loss_val, time_per_period,
                        lr, epoch, batch_counter, sample_counter)

            # Log to wandb
            Wandb.log(
                lg,
                Dict(
                    "Training/Loss" => loss,
                    "Validation/RecValLoss" => val_rec_loss,
                    "Validation/RecValLoss_min" => best_loss_val,
                    "LearningRate" => lr,
                    "Epoch" => epoch,
                    "Batch" => batch_counter,
                    "Sample" => sample_counter,
                    "TimePerPeriod" => time_per_period,
                ),
                step = sample_counter,
            )
end

function training_step!(model, x, t, β, variational, training_mode, opt)
    ps = Flux.params(model)
    loss, back = Flux.pullback(ps) do
        loss_batch(model, x, t, β, variational, training_mode)
    end
    # Backpropagate and update
    grad = back(1f0)
    Flux.Optimise.update!(opt, ps, grad)
    return loss
end

################################################################################################################
## Loss definitions

function loss_batch(model, x, t, β, variational, training_mode::SingleShooting)
    # Make prediction
    X̂, μ, logσ² = model(x, t, training_mode, variational)
    x̂, ẑ, l̂ = X̂

    # Compute reconstruction loss
    rec_loss = mean((x .- x̂).^2)/mean(abs.(x))

    # Compute KL losses for parameters and initial values
    variational && (kl_loss = vector_kl(μ, logσ²))

    return variational ? rec_loss + β * kl_loss : rec_loss
end

function loss_batch(model, x, t, β, variational, training_mode::MultipleShooting)
    # Make prediction
    X̂, μ, logσ² = model(x, t, training_mode, variational)
    x̂, ẑ, l̂ = X̂

    # Join all windows from all outputs x̂
    win_len = training_mode.win_len
    batch_size = size(x, 2)
    win_per_sample = Int(size(x̂, 2)/batch_size)
    x̂_joint = join_x(x̂, win_per_sample, win_len, batch_size)

    # Compute reconstruction loss
    rec_loss = mean((x .- x̂_joint).^2)/mean(abs.(x))

    # Compute continuity loss
    continuity_loss = cont_loss(ẑ, win_per_sample, batch_size)

    # Compute KL losses for parameters and initial values (not compatible with multiple shooting yet)
    variational && (kl_loss = vector_kl(μ, logσ²))
    C = training_mode.continuity_term

    return variational ? rec_loss + β * kl_loss + C * continuity_loss : rec_loss + C * continuity_loss
end

function evaluate_model(model, x, t, training_mode::SingleShooting)
    # Make prediction
    X̂, μ, logσ² = model(x, t, training_mode)
    x̂, ẑ, l̂ = X̂

    # Compute reconstruction loss
    rec_loss = mean((x .- x̂).^2)/mean(abs.(x))

    return rec_loss
end

function evaluate_model(model, x, t, training_mode::MultipleShooting)
    # Make prediction
    X̂, μ, logσ² = model(x, t, training_mode)
    x̂, ẑ, l̂ = X̂
    ẑ₀, θ̂ = l̂

    # Join all windows from all outputs x̂
    win_len = training_mode.win_len
    batch_size = size(x, 2)
    win_per_sample = Int(size(x̂, 2)/batch_size)
    x̂_joint = join_x(x̂, win_per_sample, win_len, batch_size)

    # Compute reconstruction loss
    rec_loss = mean((x .- x̂_joint).^2)/mean(abs.(x))

    return rec_loss
end