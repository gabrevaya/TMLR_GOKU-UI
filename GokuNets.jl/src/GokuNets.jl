module GokuNets

using SnoopPrecompile
using OrdinaryDiffEq
using StochasticDiffEq
using DiffEqNoiseProcess
using SciMLSensitivity
using Flux
using Zygote
using Statistics
using CUDA
using ModelingToolkit
import MLUtils: unstack, stack, selectdim

## Types definitions
abstract type LatentDE end

abstract type GOKU <: LatentDE end
struct GOKU_basic <: GOKU end
struct GOKU_attention <: GOKU end
abstract type TrainingMode end

struct MultipleShooting <: TrainingMode
    win_len::Int
    seq_len::Int
    continuity_term::Float32
end

struct SingleShooting <: TrainingMode
    win_len::Int
    seq_len::Int
    SingleShooting(seq_len) = new(seq_len, seq_len)
end

## Models definitions
include("./systems/hopf.jl")
export System, Hopf, Stoch_Hopf

include("./models/LatentDiffEqModel.jl")
include("./models/GOKU.jl")
include("./models/default_layers.jl")
include("./models/utils.jl")
export LatentDiffEqModel, GOKU, GOKU_basic, GOKU_attention
export default_layers
export TrainingMode, MultipleShooting, SingleShooting, mult_shooting_seq_len
export adjust_seq_len_for_multiple_shooting

SnoopPrecompile.@precompile_all_calls begin
    diffeq = Stoch_Hopf(N = 2)
    model_type = GOKU_attention()
    input_dim = 2
    seq_len = 46
    encoder_layers, decoder_layers = default_layers(model_type, input_dim, diffeq)
    model = LatentDiffEqModel(model_type, encoder_layers, decoder_layers)
    x = rand(Float32, input_dim, 64, seq_len)
    training_mode = MultipleShooting(10, seq_len, 2f0)
    t = range(0.f0, step = 0.05f0, length = training_mode.win_len)
    ps = Flux.params(model)
    X̂, μ, logσ² = model(x, t, training_mode)
end

end # module
