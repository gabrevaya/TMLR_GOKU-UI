abstract type LSTMModel end
struct LSTM_autoencoder <: LSTMModel end

import GokuNets.default_layers
import GokuNets.apply_feature_extractor
import GokuNets.apply_latent_in
import GokuNets.apply_latent_out
import GokuNets.diffeq_layer
import GokuNets.apply_reconstructor

model_name(model_type::LSTMModel, multiple_shooting) = "LSTM"

GokuNets.apply_feature_extractor(encoder::GokuNets.Encoder{T}, x) where T<:LSTM_autoencoder = encoder.feature_extractor(x)

function GokuNets.apply_pattern_extractor(encoder::GokuNets.Encoder{T}, fe_out, training_mode) where {T<:LSTM_autoencoder}
    pe_z₀ = encoder.pattern_extractor
    fe_out_unstacked = MLUtils.unstack(fe_out, dims=Val(3))
    # Reverse sequence
    fe_out_rev = reverse(fe_out_unstacked)
    pe_z₀_out = [pe_z₀(x) for x in fe_out_rev][end]
    Flux.reset!(pe_z₀)
    return pe_z₀_out
end

function GokuNets.apply_latent_in(encoder::GokuNets.Encoder{T}, pe_out) where {T<:LSTM_autoencoder}
    z̃₀ = encoder.latent_in(pe_out)
    return (z̃₀, nothing)
end

function GokuNets.apply_latent_out(decoder::GokuNets.Decoder{T}, l̃) where {T<:LSTM_autoencoder}
    ẑ₀ = decoder.latent_out(l̃)
    return ẑ₀
end

function GokuNets.diffeq_layer(decoder::GokuNets.Decoder{T}, ẑ₀, t, training_mode) where {T<:LSTM_autoencoder}
    lstm = decoder.diffeq

    ẑ_i = ẑ₀
    ẑ = ẑ₀
    for i in 1:(length(t)-1)
        ẑ_i = lstm(ẑ_i)
        ẑ = cat(ẑ, ẑ_i, dims=Val(3))
    end

    Flux.reset!(lstm)
    return ẑ
end

GokuNets.apply_reconstructor(decoder::GokuNets.Decoder{T}, ẑ) where {T<:LSTM_autoencoder} = decoder.reconstructor(ẑ)


function GokuNets.default_layers(model_type::LSTM_autoencoder, input_dim; device=cpu,
                            hidden_dim_resnet = 200, rnn_input_dim = 32, rnn_output_dim = 40,
                            latent_dim_z₀ = 16, latent_to_diffeq_dim = 200, z₀_activation = relu,
                            z_dim = 20, general_activation = relu, output_activation = σ,
                            init = Flux.kaiming_uniform(gain = 1/sqrt(3)),
                            verbose = false)

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
    pattern_extractor = pe_z₀

    # final fully connected layers before sampling
    latent_in = Dense(2*rnn_output_dim, latent_dim_z₀, init = init) |> device

    encoder_layers = (feature_extractor, pattern_extractor, latent_in)

    ######################
    ### Decoder layers ###
    ######################

    latent_out = Chain(Dense(latent_dim_z₀, latent_to_diffeq_dim, general_activation, init = init),
                    Dense(latent_to_diffeq_dim, z_dim, z₀_activation, init = init)) |> device

    LSTM_layer = LSTM(z_dim, z_dim, init = init) |> device

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

    decoder_layers = (latent_out, LSTM_layer, reconstructor)

    return encoder_layers, decoder_layers
end

function extract_model_weights(model::LatentDiffEqModel{T}) where {T<:LSTM_autoencoder}
    feature_extractor_ps = deepcopy(Flux.params(Flux.cpu(model.encoder.feature_extractor)))
    pattern_extractor_ps = deepcopy(Flux.params(Flux.cpu(model.encoder.pattern_extractor)))
    latent_in_ps = deepcopy(Flux.params(Flux.cpu(model.encoder.latent_in)))
    diffeq_ps = deepcopy(Flux.params(model.decoder.diffeq))
    latent_out_ps = deepcopy(Flux.params(Flux.cpu(model.decoder.latent_out)))
    reconstructor_ps = deepcopy(Flux.params(Flux.cpu(model.decoder.reconstructor)))
    ps_all_layers = (feature_extractor_ps, pattern_extractor_ps, latent_in_ps, diffeq_ps,
                    latent_out_ps, reconstructor_ps)
end

function load_model_weights!(model::LatentDiffEqModel{T}, ps) where {T<:LSTM_autoencoder}
    feature_extractor_ps, pattern_extractor_ps, latent_in_ps, diffeq_ps, latent_out_ps, reconstructor_ps = ps
    Flux.loadparams!(model.encoder.feature_extractor, feature_extractor_ps)
    Flux.loadparams!(model.encoder.pattern_extractor, pattern_extractor_ps)
    Flux.loadparams!(model.encoder.latent_in, latent_in_ps)
    Flux.loadparams!(model.decoder.diffeq, diffeq_ps)
    Flux.loadparams!(model.decoder.latent_out, latent_out_ps)
    Flux.loadparams!(model.decoder.reconstructor, reconstructor_ps)
end