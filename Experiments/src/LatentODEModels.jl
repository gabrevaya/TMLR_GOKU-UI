struct LatentODE <: GokuNets.LatentDE end

import GokuNets.default_layers
import GokuNets.apply_feature_extractor
import GokuNets.apply_latent_in
import GokuNets.sample
import GokuNets.apply_latent_out
import GokuNets.diffeq_layer
import GokuNets.apply_reconstructor

using DiffEqFlux
using OrdinaryDiffEq

model_name(model_type::LatentODE, multiple_shooting) = "Latent ODE"

GokuNets.apply_feature_extractor(encoder::GokuNets.Encoder{T}, x) where T<:LatentODE = encoder.feature_extractor(x)

function GokuNets.apply_pattern_extractor(encoder::GokuNets.Encoder{T}, fe_out, training_mode) where {T<:LatentODE}
    pe_z₀ = encoder.pattern_extractor
    fe_out_unstacked = MLUtils.unstack(fe_out, dims=Val(3))
    # Reverse sequence
    fe_out_rev = reverse(fe_out_unstacked)
    pe_z₀_out = [pe_z₀(x) for x in fe_out_rev][end]
    Flux.reset!(pe_z₀)
    return pe_z₀_out
end

function GokuNets.apply_latent_in(encoder::GokuNets.Encoder{T}, pe_out) where {T<:LatentODE}
    li_μ_z₀, li_logσ²_z₀ = encoder.latent_in

    z̃₀_μ = li_μ_z₀(pe_out)
    z̃₀_logσ² = li_logσ²_z₀(pe_out)

    return z̃₀_μ, z̃₀_logσ²
end

function GokuNets.sample(μ::P, logσ²::P, model::LatentDiffEqModel{T}) where {P<:Array, T<:LatentODE}
    ẑ₀ = μ + randn(Float32, size(logσ²)) .* exp.(logσ²/2f0)
    return ẑ₀
end

function sample(μ::P, logσ²::P, model::LatentDiffEqModel{T}) where {P<:Flux.CUDA.CuArray, T<:LatentODE}
    ẑ₀ = μ + gpu(randn(Float32, size(logσ²))) .* exp.(logσ²/2f0)
    return ẑ₀
end

function GokuNets.apply_latent_out(decoder::GokuNets.Decoder{T}, l̃) where {T<:LatentODE}
    ẑ₀ = decoder.latent_out(l̃)
    return ẑ₀
end

function GokuNets.diffeq_layer(decoder::GokuNets.Decoder{T}, ẑ₀, t, training_mode) where {T<:LatentODE}
    node = decoder.diffeq 
    sol = node(ẑ₀)
    ẑ = to_device(ẑ₀, sol)
    return ẑ
end

to_device(x::CuArray, y) = gpu(y)
to_device(x::Array, y) = Array(y)

GokuNets.apply_reconstructor(decoder::GokuNets.Decoder{T}, ẑ) where {T<:LatentODE} = decoder.reconstructor(ẑ)

function vector_kl(μ::T, logσ²::T) where {T <: CuArray}
	return mean(sum(kl.(μ, logσ²), dims=1))
end

function GokuNets.default_layers(model_type::LatentODE, input_dim, diffeq; device=cpu,
                            hidden_dim_resnet = 200, rnn_input_dim = 32, rnn_output_dim = 40,
                            latent_dim_z₀ = 16, latent_to_diffeq_dim = 200, z_dim = 20, augment_dim = 10,
                            z₀_activation = relu, general_activation = relu, output_activation = σ,
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
    li_μ_z₀ = Dense(rnn_output_dim*2, latent_dim_z₀, init = init) |> device
    li_logσ²_z₀ = Dense(rnn_output_dim*2, latent_dim_z₀, init = init) |> device
    latent_in = (li_μ_z₀, li_logσ²_z₀)

    encoder_layers = (feature_extractor, pattern_extractor, latent_in)

    ######################
    ### Decoder layers ###
    ######################

    latent_out = Chain(Dense(latent_dim_z₀, latent_to_diffeq_dim, general_activation, init = init),
                    Dense(latent_to_diffeq_dim, z_dim, z₀_activation, init = init)) |> device

    # going back to the input space
    # Resnet
    l1 = Dense(z_dim + augment_dim, hidden_dim_resnet, general_activation, init = init)
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

function extract_model_weights(model::LatentDiffEqModel{T}) where {T<:LatentODE}
    feature_extractor_ps = deepcopy(Flux.params(Flux.cpu(model.encoder.feature_extractor)))
    pattern_extractor_ps = deepcopy(Flux.params(Flux.cpu(model.encoder.pattern_extractor)))
    latent_in_ps = deepcopy(Flux.params(Flux.cpu(model.encoder.latent_in)))
    node_ps = deepcopy(Flux.params(model.decoder.diffeq.model))
    latent_out_ps = deepcopy(Flux.params(Flux.cpu(model.decoder.latent_out)))
    reconstructor_ps = deepcopy(Flux.params(Flux.cpu(model.decoder.reconstructor)))
    ps_all_layers = (feature_extractor_ps, pattern_extractor_ps, latent_in_ps, node_ps,
                    latent_out_ps, reconstructor_ps)
end

function load_model_weights!(model::LatentDiffEqModel{T}, ps) where {T<:LatentODE}
    feature_extractor_ps, pattern_extractor_ps, latent_in_ps, node_ps, latent_out_ps, reconstructor_ps = ps
    Flux.loadparams!(model.encoder.feature_extractor, feature_extractor_ps)
    Flux.loadparams!(model.encoder.pattern_extractor, pattern_extractor_ps)
    Flux.loadparams!(model.encoder.latent_in, latent_in_ps)
    Flux.loadparams!(model.decoder.diffeq.model, node_ps)
    Flux.loadparams!(model.decoder.latent_out, latent_out_ps)
    Flux.loadparams!(model.decoder.reconstructor, reconstructor_ps)
end