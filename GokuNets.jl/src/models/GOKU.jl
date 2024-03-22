# GOKU-net model
#
# GOKU_basic model based on
# https://arxiv.org/abs/2003.10775

apply_feature_extractor(encoder::Encoder{T}, x) where {T<:GOKU} = encoder.feature_extractor(x)

function apply_pattern_extractor(encoder::Encoder{T}, fe_out, training_mode::SingleShooting) where {T<:GOKU}
    pe_z₀, pe_θ_forward, pe_θ_backward = encoder.pattern_extractor
    fe_out = unstack(fe_out, dims=Val(3))
    
    # reverse sequence
    fe_out_rev = reverse(fe_out)

    # pass it through the recurrent layers
    pe_z₀_out = [pe_z₀(x) for x in fe_out_rev][end]
    pe_θ_out_f = [pe_θ_forward(x) for x in fe_out][end]
    pe_θ_out_b = [pe_θ_backward(x) for x in fe_out_rev][end]
    pe_θ_out = vcat(pe_θ_out_f, pe_θ_out_b)

    # reset hidden states
    Flux.reset!(pe_z₀)
    Flux.reset!(pe_θ_forward)
    Flux.reset!(pe_θ_backward)

    return pe_z₀_out, pe_θ_out
end

function apply_pattern_extractor(encoder::Encoder{GOKU_attention}, fe_out, training_mode::SingleShooting)
    pe_z₀, pe_θ_forward, pe_θ_backward, pe_θ_attn = encoder.pattern_extractor
    
    fe_out = unstack(fe_out, dims=Val(3))

    # Reverse sequence
    fe_out_rev = reverse(fe_out)

    # Pass it through the recurrent layers
    pe_z₀_out = [pe_z₀(x) for x in fe_out_rev][end]
    pe_θ_out_f = [pe_θ_forward(x) for x in fe_out]
    pe_θ_out_b = [pe_θ_backward(x) for x in fe_out_rev]

    pe_θ_out = vcat.(pe_θ_out_f, pe_θ_out_b)
    pe_θ_out = stack(pe_θ_out, dims=3)

    # Attention for the biLSTM, for the params of the diff eq
    θ_attn_scores = softmax(pe_θ_attn(pe_θ_out), dims=3)
    pe_θ_out = dropdims(sum(pe_θ_out.*θ_attn_scores, dims=3), dims=3)

    # Reset hidden states
    Flux.reset!(pe_z₀)
    Flux.reset!(pe_θ_forward)
    Flux.reset!(pe_θ_backward)

    return pe_z₀_out, pe_θ_out
end

function apply_pattern_extractor(encoder::Encoder{GOKU_attention}, fe_out, training_mode::MultipleShooting)
    pe_z₀, pe_θ_forward, pe_θ_backward, pe_θ_attn = encoder.pattern_extractor
    win_len = training_mode.win_len
    seq_len = training_mode.seq_len

    # Split the each sample of each batch into segments for multiple shooting
    # and arrange them so that an initial condition is inferred for each of them
    fe_out_splitted = split_for_multiple_shooting(fe_out, seq_len, win_len)
    fe_out_splitted = unstack(fe_out_splitted, dims=Val(2))
    fe_out = unstack(fe_out, dims=Val(3))

    # Reverse sequence
    fe_out_rev = reverse(fe_out)
    fe_out_splitted_rev = reverse(fe_out_splitted)

    # Pass it through the recurrent layers
    pe_z₀_out = [pe_z₀(x) for x in fe_out_splitted_rev][end]
    pe_θ_out_f = [pe_θ_forward(x) for x in fe_out]
    pe_θ_out_b = [pe_θ_backward(x) for x in fe_out_rev]

    pe_θ_out = vcat.(pe_θ_out_f, pe_θ_out_b)
    pe_θ_out = Flux.stack(pe_θ_out, dims=3)

    # Attention for the biLSTM, for the params of the diff eq
    θ_attn_scores = softmax(pe_θ_attn(pe_θ_out), dims=3)
    pe_θ_out = dropdims(sum(pe_θ_out.*θ_attn_scores, dims=3), dims=3)

    # Reset hidden states
    Flux.reset!(pe_z₀)
    Flux.reset!(pe_θ_forward)
    Flux.reset!(pe_θ_backward)

    return pe_z₀_out, pe_θ_out
end

function apply_pattern_extractor(encoder::Encoder{T}, fe_out, training_mode::MultipleShooting) where {T<:GOKU}
    pe_z₀, pe_θ_forward, pe_θ_backward = encoder.pattern_extractor
    win_len = training_mode.win_len
    seq_len = training_mode.seq_len

    # Split the each sample of each batch into segments for multiple shooting
    # and arrange them so that an initial condition is inferred for each of them
    fe_out_splitted = split_for_multiple_shooting(fe_out, seq_len, win_len)
    fe_out_splitted = unstack(fe_out_splitted, dims=Val(2))
    fe_out = unstack(fe_out, dims=Val(3))

    # Reverse sequence
    fe_out_rev = reverse(fe_out)
    fe_out_splitted_rev = reverse(fe_out_splitted)

    # pass it through the recurrent layers
    pe_z₀_out = [pe_z₀(x) for x in fe_out_splitted_rev][end]
    pe_θ_out_f = [pe_θ_forward(x) for x in fe_out][end]
    pe_θ_out_b = [pe_θ_backward(x) for x in fe_out_rev][end]
    pe_θ_out = vcat(pe_θ_out_f, pe_θ_out_b)

    # reset hidden states
    Flux.reset!(pe_z₀)
    Flux.reset!(pe_θ_forward)
    Flux.reset!(pe_θ_backward)

    return pe_z₀_out, pe_θ_out
end

function apply_latent_in(encoder::Encoder{T}, pe_out) where {T<:GOKU}
    pe_z₀_out, pe_θ_out = pe_out
    li_μ_z₀, li_logσ²_z₀, li_μ_θ, li_logσ²_θ = encoder.latent_in

    z₀_μ = li_μ_z₀(pe_z₀_out)
    z₀_logσ² = li_logσ²_z₀(pe_z₀_out)

    θ_μ = li_μ_θ(pe_θ_out)
    θ_logσ² = li_logσ²_θ(pe_θ_out)

    return (z₀_μ, θ_μ), (z₀_logσ², θ_logσ²)
end

function apply_latent_out(decoder::Decoder{T}, l̃) where {T<:GOKU}
    z̃₀, θ̃ = l̃
    lo_z₀, lo_θ = decoder.latent_out

    ẑ₀ = lo_z₀(z̃₀)
    θ̂ = lo_θ(θ̃)

    return ẑ₀, θ̂
end

function diffeq_layer(decoder::Decoder{T}, l̂, t, training_mode) where {T<:GOKU}
    ẑ₀_, θ̂_ = l̂
    prob = decoder.diffeq.prob
    solver = decoder.diffeq.solver
    sensealg = decoder.diffeq.sensealg
    kwargs = decoder.diffeq.kwargs
    
    # Make sure the diff eq  solving is done on cpu
    ẑ₀ = convert(Matrix{Float32}, ẑ₀_)
    θ̂ = convert(Matrix{Float32}, θ̂_)
 
    ẑ = ensemble_solve(prob, ẑ₀, θ̂, t, solver, sensealg, training_mode)

    # Optionally transform the latent state variables
    ẑ = transform_after_diffeq(Array(ẑ), decoder.diffeq)
    ẑ = permutedims(ẑ, [1,3,2])
    # ẑ = device(ẑ₀_, ẑ)
    return ẑ
end

function ensemble_solve(prob, z₀, θ, t, solver, sensealg, training_mode)
    prob_func = get_prob_func(prob, z₀, θ, training_mode)
    output_func(sol, i) = sol.retcode == SciMLBase.ReturnCode.Success ? (Array(sol), false) : (fill(NaN32,(size(z₀, 1), length(t))), false)
    prob = remake(prob; tspan = (t[1], t[end]))
    ens_prob = EnsembleProblem(prob, prob_func = prob_func, output_func = output_func, safetycopy=false)
    sol = solve(ens_prob, solver, EnsembleThreads(); sensealg = sensealg, trajectories = size(z₀, 2), saveat = t)
    Array(sol)
end

function get_prob_func(prob, z₀, θ, training_mode::MultipleShooting)
    ind(x) = Int((x - 1) ÷ (size(z₀, 2) / size(θ, 2)) + 1)
    prob_func(prob, i, repeat) = remake(prob, u0=z₀[:, i], p = θ[:, ind(i)])
    return prob_func
end

function get_prob_func(prob, z₀, θ, training_mode::SingleShooting)
    prob_func(prob, i, repeat) = remake(prob, u0= z₀[:, i], p = θ[:, i])
    return prob_func
end

device(x::Flux.CUDA.CuArray, y) = gpu(y)
device(x::Array, y) = y

# Identity by default
transform_after_diffeq(x, diffeq) = x

apply_reconstructor(decoder::Decoder{T}, ẑ) where {T<:GOKU} = decoder.reconstructor(ẑ)

function sample(μ::P, logσ²::P, model::LatentDiffEqModel{T}) where {P<:Tuple{Array, Array}, T<:GOKU}
    z₀_μ, θ_μ = μ
    z₀_logσ², θ_logσ² = logσ²

    ẑ₀ = z₀_μ + randn(Float32, size(z₀_logσ²)) .* exp.(z₀_logσ²/2f0)
    θ̂ =  θ_μ + randn(Float32, size( θ_logσ²)) .* exp.(θ_logσ²/2f0)

    return ẑ₀, θ̂
end

function sample(μ::P, logσ²::P, model::LatentDiffEqModel{T}) where {P<:Tuple{Flux.CUDA.CuArray, Flux.CUDA.CuArray}, T<:GOKU}
    z₀_μ, θ_μ = μ
    z₀_logσ², θ_logσ² = logσ²

    ẑ₀ = z₀_μ + gpu(randn(Float32, size(z₀_logσ²))) .* exp.(z₀_logσ²/2f0)
    θ̂ =  θ_μ + gpu(randn(Float32, size( θ_logσ²))) .* exp.(θ_logσ²/2f0)

    return ẑ₀, θ̂
end

