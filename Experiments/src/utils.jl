
################################################################################
## Loss utility functions

function vector_mse(x, x̂)
    res = zero(eltype(x[1]))
    @inbounds for i in eachindex(x)
        res += sum((x[i] .- x̂[i]).^2)
    end
    # divide per number of time steps and batch size
    res /= size(x, 1) * size(x[1], 2)
    return res
end

# KL divergence
kl(μ, logσ²) = (exp(logσ²) + μ^2 - logσ² - 1) / 2

function vector_kl(μ::T, logσ²::T) where T <: Tuple{Matrix, Matrix}
    P = eltype(μ[1])
    s = zero(P)
    # go through initial conditions and parameters
    @inbounds for i in 1:2
        s1 = zero(P)
        @inbounds for k in eachindex(μ[i])
            s1 += kl(μ[i][k], logσ²[i][k])
        end
        # divide per batch size
        s1 /= size(μ[i], 2)
        s += s1
    end
    return s
end

function vector_kl(μ::T, logσ²::T) where T <: Matrix
    P = eltype(μ)
    s = zero(P)
    # go through initial conditions
    @inbounds for k in eachindex(μ)
        s += kl(μ[k], logσ²[k])
    end
    # divide per batch size
    s /= size(μ, 2)
    return s
end

function vector_kl(μ::T, logσ²::T) where T <: Tuple{CuArray, CuArray}
	return sum( [ mean(sum(kl.(μ[i], logσ²[i]), dims=1)) for i in 1:length(μ) ] )
end

# Define a function that returns non-multiples of `k` in the range 1 to `N`.
non_multiples(N::Integer, k::Integer) = filter(x -> x % k != 0, 1:N)

function join_x(x̂, win_per_sample::Integer, win_len::Integer, batch_size::Integer)
    indxs = non_multiples(win_per_sample*batch_size, win_per_sample)
    win_minus_1 = win_per_sample - 1
    x_all = x̂[:, indxs, 1:(win_len - 1)]
    x_reshaped = reshape(x_all, size(x_all, 1), win_minus_1, batch_size, size(x_all, 3))
    x_reordered = permutedims(x_reshaped, (1, 3, 4, 2))
    x_joint = reshape(x_reordered, size(x_all, 1), batch_size, win_minus_1 * size(x_all, 3))
    last_steps = x̂[:, win_per_sample:win_per_sample:end, :];
    x_joint = cat(x_joint, last_steps, dims=Val(3))
end

function cont_loss(z, win_per_sample::Integer, batch_size::Integer)
    inds = non_multiples(win_per_sample*batch_size, win_per_sample)
    z_end = @view z[:, inds, end]
    z_begin = @view z[:, inds .+ 1, 1]
    s = sum((z_end .- z_begin) .^ 2)
    return s / (size(z, 1) * (win_per_sample - 1) * batch_size)
end

## annealing factor scheduler
# based on https://github.com/haofuml/cyclical_annealing
function frange_cycle_linear(n_iter, start::T=0.0f0, stop::T=1.0f0,  n_cycle=4, ratio=0.5) where T
    L = ones(n_iter) * stop
    period = n_iter/n_cycle
    step = T((stop-start)/(period*ratio)) # linear schedule

    for c in 0:n_cycle-1
        v, i = start, 1
        while (v ≤ stop) & (Int(round(i+c*period)) < n_iter)
            L[Int(round(i+c*period))] = v
            v += step
            i += 1
        end
    end
    return T.(L)
end

################################################################################
## Data pre-processing

function normalize_to_unit_segment(X)
    min_val = minimum(X)
    max_val = maximum(X)

    X̂ = (X .- min_val) ./ (max_val - min_val)
    return X̂, min_val, max_val
end

denormalize_unit_segment(X̂, min_val, max_val) = X̂ .* (max_val .- min_val) .+ min_val


################################################################################
## Training help function

function time_loader(x, full_seq_len, seq_len)

    x_ = Array{Float32, 3}(undef, (size(x,1), size(x,2), seq_len))
    idxs = rand_time(full_seq_len, seq_len)
    for i in 1:size(x,2)
        x_[:,i,:] = x[:, i, idxs]
    end
    return x_
end

function rand_time(full_seq_len, seq_len)
    start_time = rand(1:full_seq_len - seq_len)
    idxs = start_time:start_time+seq_len-1
    return idxs
end

function time_loader(x, z, full_seq_len, seq_len)
    x_ = Array{Float32, 3}(undef, (size(x,1), size(x,2), seq_len))
    z_ = Array{Float32, 3}(undef, (size(z,1), size(z,2), seq_len))
    idxs = rand_time(full_seq_len, seq_len)
    for i in 1:size(x,2)
        x_[:,i,:] = x[:, i, idxs]
        z_[:,i,:] = z[:, i, idxs]
    end
    return x_, z_
end

elapsed_time(t0) = Float32(round((time_ns() - t0)/1e9, digits=2))

function show_time(progress)
    elapsed_time = round(Int, progress.tlast - progress.tinit)
    return string(elapsed_time)
end

function convert_from_seconds(sec::Int)
    x, seconds = divrem(sec, 60)
    hours, minutes = divrem(x, 60)
    hours, minutes, seconds
end

function early_stop_info(training_state)
    @info "Early stopped."
    @info "Epochs: $(training_state.epoch)"
    @info "Training batches: $(training_state.batch_counter)"
end

function finishing_info(training_state, progress)
    hours, minutes, seconds = convert_from_seconds(round(Int, time() - progress.tinit))
    @info "Elapsed time: $hours:$minutes:$seconds"
    @info "Lowest validation loss: $(training_state.best_loss_val)"
end

# plt = pyimport("matplotlib.pyplot")

function plot_for_wandb(z, ẑ)

    plt = pyimport("matplotlib.pyplot")
    N = Int(size(z, 1)/2)
    fig, axs = plt.subplots(2, N)
    for i in 1:N
        axs[1,i].plot(z[i,:], label = "true")
        axs[1,i].plot(ẑ[i,:], label = "model")

        axs[2,i].plot(z[i+N,:], label = "true")
        axs[2,i].plot(ẑ[i+N,:], label = "predicted")

        axs[2,i].set_xlabel("time steps")
        axs[i].grid(true)
        axs[i+N].grid(true)

        axs[i+N].label_outer()
        axs[i].label_outer()
    end
    plt.close()
    return fig, axs
end

# function plot_high_dims_for_wandb(x, x̂, w, h, data_source::fMRI_Polo)

#     # N = Int(size(x, 1)/2)
#     plt = pyimport("matplotlib.pyplot")
#     N = 6
#     fig, axs = plt.subplots(2, N)
#     for i in 1:N-1 #N
#         axs[1,i].plot(x[i,:], label = "true")
#         axs[1,i].plot(x̂[i,:], label = "model")

#         axs[2,i].plot(x[i+N,:], label = "true")
#         axs[2,i].plot(x̂[i+N,:], label = "predicted")

#         axs[2,i].set_xlabel("time steps")
#         axs[i].grid(true)
#         axs[i+N].grid(true)

#         axs[i+N].label_outer()
#         axs[i].label_outer()
#     end
#     #specific for odd number of comps
#     axs[1,N].plot(x[N,:], label = "true")
#     axs[1,N].plot(x̂[N,:], label = "model")

#     axs[2,N].set_xlabel("time steps")
#     axs[N].grid(true)
#     axs[N].label_outer()
#     plt.close()
#     return fig, axs
# end

function plot_high_dims_for_wandb2(x, x̂, w, h)
    plt = pyimport("matplotlib.pyplot")
    fig = plt.figure()

    axs = fig.add_subplot(2, 1, 1)
    imgplot = plt.imshow(transpose(x), cmap="jet", )
    plt.axis("off")
    plt.gca().set_title("true") 

    axs = fig.add_subplot(2, 1, 2)
    imgplot = plt.imshow(transpose(x̂), cmap="jet")
    plt.axis("off")
    plt.gca().set_title("model")
    plt.close()
    return fig, axs
end

# Adaptation of SinExp from ParameterSchedulers for having wider minimums of lr
import ParameterSchedulers.AbstractSchedule
"""
    Sin4(range, offset, period)
    Sin4(;λ0, λ1, period)
A sine wave schedule with `period`.
The output conforms to
```text
abs(λ0 - λ1) * (sin(π * (t - 1) / period))^4 + min(λ0, λ1)
```
# Arguments
- `range == abs(λ0 - λ1)`: the dynamic range (given by the endpoints)
- `offset == min(λ0, λ1)`: the offset / minimum value
- `period::Integer`: the period
"""
struct Sin4{T, S<:Integer} <: AbstractSchedule{false}
    range::T
    offset::T
    period::S
end
function Sin4(range::T, offset::T, period::S) where {T, S}
    Sin4{T, S}(range, offset, period)
end
Sin4(;λ0, λ1, period) = Sin4(abs(λ0 - λ1), min(λ0, λ1), period)

Base.eltype(::Type{<:Sin4{T}}) where T = T

(schedule::Sin4)(t) = schedule.range * _sin4(t, schedule.period) + schedule.offset

_sin4(t, period) = sin(π * (t - 1) / period)^4 # for having wider minimums of lr

"""
    Sin4Exp(range, offset, period, γ)
    Sin4Exp(;λ0, λ1, period, γ)
A sine wave schedule with `period` and an exponentially decaying amplitude.
The output conforms to
```text
abs(λ0 - λ1) * Sin⁴(t) * γ^(t - 1) + min(λ0, λ1)
```
where `Sin⁴(t)` is `sin(π * (t - 1) / period)^4`.
# Arguments
- `range == abs(λ0 - λ1)`: the dynamic range (given by the endpoints)
- `offset == min(λ0, λ1)`: the offset / minimum value
- `period::Integer`: the period
- `γ`: the decay rate
"""
_sin4exp(range, offset, period, γ) =
    ComposedSchedule(Sin4(range, offset, period), (Exp(range, γ), offset, period))
function Sin4Exp(range, offset, period, γ)
    return _sin4exp(range, offset, period, γ)
end
Sin4Exp(;λ0, λ1, period, γ) = _sin4exp(abs(λ0 - λ1), min(λ0, λ1), period, γ)





# Adaptation of SinExp from ParameterSchedulers for cos and having wider minimums of lr
import ParameterSchedulers.AbstractSchedule
"""
    Cos4(range, offset, period)
    Cos4(;λ0, λ1, period)
A sine wave schedule with `period`.
The output conforms to
```text
abs(λ0 - λ1) * (cos(π * (t - 1) / period))^4 + min(λ0, λ1)
```
# Arguments
- `range == abs(λ0 - λ1)`: the dynamic range (given by the endpoints)
- `offset == min(λ0, λ1)`: the offset / minimum value
- `period::Integer`: the period
"""
struct Cos4{T, S<:Integer} <: AbstractSchedule{false}
    range::T
    offset::T
    period::S
end
function Cos4(range::T, offset::T, period::S) where {T, S}
    Cos4{T, S}(range, offset, period)
end
Cos4(;λ0, λ1, period) = Cos4(abs(λ0 - λ1), min(λ0, λ1), period)

Base.eltype(::Type{<:Cos4{T}}) where T = T

(schedule::Cos4)(t) = schedule.range * _cos4(t, schedule.period) + schedule.offset

_cos4(t, period) = cos(π * (t - 1) / period)^4 # for having wider minimums of lr

"""
    Cos4Exp(range, offset, period, γ)
    Cos4Exp(;λ0, λ1, period, γ)
A sine wave schedule with `period` and an exponentially decaying amplitude.
The output conforms to
```text
abs(λ0 - λ1) * Cos⁴(t) * γ^(t - 1) + min(λ0, λ1)
```
where `Cos⁴(t)` is `cos(π * (t - 1) / period)^4`.
# Arguments
- `range == abs(λ0 - λ1)`: the dynamic range (given by the endpoints)
- `offset == min(λ0, λ1)`: the offset / minimum value
- `period::Integer`: the period
- `γ`: the decay rate
"""
_cos4exp(range, offset, period, γ) =
    ComposedSchedule(Cos4(range, offset, period), (Exp(range, γ), offset, period))
function Cos4Exp(range, offset, period, γ)
    return _cos4exp(range, offset, period, γ)
end
Cos4Exp(;λ0, λ1, period, γ) = _cos4exp(abs(λ0 - λ1), min(λ0, λ1), period, γ)

function describe(diffeq::GokuNets.System)
    diffeq_type = diffeq |> typeof |> nameof
    diffeq_dim = length(diffeq.prob.u0)
    diffeq_state_size = length(diffeq.prob.u0)
    diffeq_p_size = length(diffeq.prob.p)
    return diffeq_type, diffeq_dim, diffeq_state_size, diffeq_p_size
end

# Functions for saving and loading model weights
function save_model_weights(model, out_dir, file_name)
    feature_extractor_ps = Flux.params(Flux.cpu(model.encoder.feature_extractor))
    pattern_extractor_ps = Flux.params(Flux.cpu(model.encoder.pattern_extractor))
    latent_in_ps = Flux.params(Flux.cpu(model.encoder.latent_in))
    latent_out_ps = Flux.params(Flux.cpu(model.decoder.latent_out))
    reconstructor_ps = Flux.params(Flux.cpu(model.decoder.reconstructor))
    ps_all_layers = (feature_extractor_ps, pattern_extractor_ps, latent_in_ps, latent_out_ps, reconstructor_ps)
    JLD2.@save "$out_dir/$file_name.jld2" ps_all_layers
end

function extract_model_weights(model)
    feature_extractor_ps = deepcopy(Flux.params(Flux.cpu(model.encoder.feature_extractor)))
    pattern_extractor_ps = deepcopy(Flux.params(Flux.cpu(model.encoder.pattern_extractor)))
    latent_in_ps = deepcopy(Flux.params(Flux.cpu(model.encoder.latent_in)))
    latent_out_ps = deepcopy(Flux.params(Flux.cpu(model.decoder.latent_out)))
    reconstructor_ps = deepcopy(Flux.params(Flux.cpu(model.decoder.reconstructor)))
    ps_all_layers = (feature_extractor_ps, pattern_extractor_ps, latent_in_ps, latent_out_ps, reconstructor_ps)
end

function load_model_weights!(model, ps)
    feature_extractor_ps, pattern_extractor_ps, latent_in_ps, latent_out_ps, reconstructor_ps = ps
    Flux.loadparams!(model.encoder.feature_extractor, feature_extractor_ps)
    Flux.loadparams!(model.encoder.pattern_extractor, pattern_extractor_ps)
    Flux.loadparams!(model.encoder.latent_in, latent_in_ps)
    Flux.loadparams!(model.decoder.latent_out, latent_out_ps)
    Flux.loadparams!(model.decoder.reconstructor, reconstructor_ps)
end

function get_file_path(config, n_traj=1000)
    @unpack data_path = config
  
    if split(data_path, ".")[end] == "h5"
        return data_path
    else
        mkpath(data_path)
        @unpack diffeq, diffeq_args = config
        diffeq = diffeq(;diffeq_args...)
        diffeq_type = diffeq |> typeof |> nameof
        diffeq_dim = length(diffeq.prob.u0)
        filename = "data_$(diffeq_type)_$(round(Int, diffeq_dim/2))_samples=$n_traj.h5"
        return joinpath(data_path, filename)
    end
end
  