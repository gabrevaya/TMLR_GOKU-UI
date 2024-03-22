function default_layers(model_type::GOKU_basic, input_dim, diffeq; device=cpu,
                            hidden_dim_resnet = 200, rnn_input_dim = 32,
                            rnn_output_dim = 16, latent_dim_z₀ = 16, latent_dim_θ = 16,
                            latent_to_diffeq_dim = 200, general_activation = relu,
                            z₀_activation = identity, θ_activation = softplus,
                            output_activation = σ, init = Flux.kaiming_uniform(gain = 1/sqrt(3)),
                            verbose = false)

    z_dim = length(diffeq.prob.u0)
    θ_dim = length(diffeq.prob.p)

    ######################
    ### Encoder layers ###
    ######################
    # Resnet
    l1 = Dense(input_dim, hidden_dim_resnet, general_activation, init = init)
    l2 = Dense(hidden_dim_resnet, hidden_dim_resnet, general_activation, init = init)
    l3 = Dense(hidden_dim_resnet, hidden_dim_resnet, general_activation, init = init)
    l4 = Dense(hidden_dim_resnet, rnn_input_dim, general_activation, init = init)
    feature_extractor = Chain(l1,
                                SkipConnection(l2, +),
                                SkipConnection(l3, +),
                                l4) |> device

    # RNN
    pe_z₀ = Chain(RNN(rnn_input_dim, rnn_output_dim, relu, init = init),
                  RNN(rnn_output_dim, rnn_output_dim, relu, init = init)) |> device

    # Bidirectional LSTM
    pe_θ_forward = Chain(LSTM(rnn_input_dim, rnn_output_dim, init = init),
                         LSTM(rnn_output_dim, rnn_output_dim, init = init)) |> device

    pe_θ_backward = Chain(LSTM(rnn_input_dim, rnn_output_dim, init = init),
                          LSTM(rnn_output_dim, rnn_output_dim, init = init)) |> device

    pattern_extractor = (pe_z₀, pe_θ_forward, pe_θ_backward)

    # final fully connected layers before sampling
    li_μ_z₀ = Dense(rnn_output_dim, latent_dim_z₀, init = init) |> device
    li_logσ²_z₀ = Dense(rnn_output_dim, latent_dim_z₀, init = init) |> device

    li_μ_θ = Dense(rnn_output_dim*2, latent_dim_θ, init = init) |> device
    li_logσ²_θ = Dense(rnn_output_dim*2, latent_dim_θ, init = init) |> device

    latent_in = (li_μ_z₀, li_logσ²_z₀, li_μ_θ, li_logσ²_θ)

    encoder_layers = (feature_extractor, pattern_extractor, latent_in)

    ######################
    ### Decoder layers ###
    ######################

    # after sampling in the latent space but before the differential equation layer
    lo_z₀ = Chain(Dense(latent_dim_z₀, latent_to_diffeq_dim, general_activation, init = init),
                  Dense(latent_to_diffeq_dim, z_dim, z₀_activation, init = init)) |> device

    lo_θ = Chain(Dense(latent_dim_θ, latent_to_diffeq_dim, general_activation, init = init),
                 Dense(latent_to_diffeq_dim, θ_dim, θ_activation, init = init)) |> device

    latent_out = (lo_z₀, lo_θ)

    # going back to the input space
    # Resnet
    l1 = Dense(z_dim, hidden_dim_resnet, general_activation, init = init)
    l2 = Dense(hidden_dim_resnet, hidden_dim_resnet, general_activation, init = init)
    l3 = Dense(hidden_dim_resnet, hidden_dim_resnet, general_activation, init = init)
    l4 = Dense(hidden_dim_resnet, input_dim, output_activation, init = init)
    reconstructor = Chain(l1,
                            SkipConnection(l2, +),
                            SkipConnection(l3, +),
                            l4)  |> device

    decoder_layers = (latent_out, diffeq, reconstructor)

    return encoder_layers, decoder_layers
end

function default_layers(model_type::GOKU_attention, input_dim, diffeq; device=cpu,
                            hidden_dim_resnet = 200, rnn_input_dim = 32,
                            rnn_output_dim = 40, latent_dim_z₀ = 16, latent_dim_θ = 16,
                            latent_to_diffeq_dim = 200, general_activation = relu,
                            z₀_activation = x->x, θ_activation = softplus,
                            output_activation = σ, init = Flux.kaiming_uniform(gain = 1/sqrt(3)),
                            verbose = false)

    verbose && @info "Using custom layers"
    z_dim = length(diffeq.prob.u0)
    θ_dim = length(diffeq.prob.p)

    ######################
    ### Encoder layers ###
    ######################
    # Resnet
    l1 = Dense(input_dim, hidden_dim_resnet, general_activation, init = init)
    l2 = Dense(hidden_dim_resnet, hidden_dim_resnet, general_activation, init = init)
    l3 = Dense(hidden_dim_resnet, hidden_dim_resnet, general_activation, init = init)
    l4 = Dense(hidden_dim_resnet, rnn_input_dim, general_activation, init = init)
    feature_extractor = Chain(l1,
                                SkipConnection(l2, +),
                                SkipConnection(l3, +),
                                l4) |> device
    # recurrent pattern extractor

    pe_z₀ = LSTM(rnn_input_dim, 2*rnn_output_dim, init = init) |> device

    pe_θ_forward = LSTM(rnn_input_dim, rnn_output_dim, init = init) |> device
    pe_θ_backward = LSTM(rnn_input_dim, rnn_output_dim, init = init) |> device  

    # pe_z₀_attn = Dense(2*rnn_output_dim, 2*rnn_output_dim) |> device
    pe_θ_attn = Dense(2*rnn_output_dim, 2*rnn_output_dim) |> device

    # pattern_extractor = (pe_z₀, pe_θ_forward, pe_θ_backward, pe_z₀_attn, pe_θ_attn)
    pattern_extractor = (pe_z₀, pe_θ_forward, pe_θ_backward, pe_θ_attn)

    # final fully connected layers before sampling
    li_μ_z₀ = Dense(rnn_output_dim*2, latent_dim_z₀, init = init) |> device
    li_logσ²_z₀ = Dense(rnn_output_dim*2, latent_dim_z₀, init = init) |> device

    li_μ_θ = Dense(rnn_output_dim*2, latent_dim_θ, init = init) |> device
    li_logσ²_θ = Dense(rnn_output_dim*2, latent_dim_θ, init = init) |> device

    latent_in = (li_μ_z₀, li_logσ²_z₀, li_μ_θ, li_logσ²_θ)

    encoder_layers = (feature_extractor, pattern_extractor, latent_in)

    ######################
    ### Decoder layers ###
    ######################

    # after sampling in the latent space but before the differential equation layer
    lo_z₀ = Chain(Dense(latent_dim_z₀, latent_to_diffeq_dim, general_activation, init = init),
    Dense(latent_to_diffeq_dim, z_dim, z₀_activation, init = init)) |> device

    lo_θ = Chain(Dense(latent_dim_θ, latent_to_diffeq_dim, general_activation, init = init),
    Dense(latent_to_diffeq_dim, θ_dim, θ_activation, init = init)) |> device

    latent_out = (lo_z₀, lo_θ)

    # going back to the input space
    # Resnet
    l1 = Dense(z_dim, hidden_dim_resnet, general_activation, init = init)
    l2 = Dense(hidden_dim_resnet, hidden_dim_resnet, general_activation, init = init)
    l3 = Dense(hidden_dim_resnet, hidden_dim_resnet, general_activation, init = init)
    l4 = Dense(hidden_dim_resnet, input_dim, output_activation, init = init)
    reconstructor = Chain(l1,
                            SkipConnection(l2, +),
                            SkipConnection(l3, +),
                            l4)  |> device

    decoder_layers = (latent_out, diffeq, reconstructor)

    return encoder_layers, decoder_layers
end