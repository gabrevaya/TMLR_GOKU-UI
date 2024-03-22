module TMLR_experiments

using GokuNets

using OrdinaryDiffEq
using StochasticDiffEq
using ModelingToolkit
using SciMLSensitivity

using Flux
import Flux: Chain
using Zygote
using MLUtils
using ParameterSchedulers
using CUDA
CUDA.allowscalar(false)

using DrWatson
using Random
using Statistics
using Distributions
using Dates
using Logging
using Wandb
using PyCall
using ProgressMeter: Progress, next!, durationstring
using Parameters
using Distributed
using SlurmClusterManager
using FileIO
using JLD2
using HDF5

using DataFrames
import DataFrames: combine
using DataFramesMeta
using HypothesisTests
using MultipleTesting
using Plots
using ColorSchemes

# Include code
include(srcdir("data_generation.jl"))
include(srcdir("default_data_gen_args.jl"))
include(srcdir("dataloaders.jl"))
include(srcdir("utils.jl"))
include(srcdir("farm_utils.jl"))
include(srcdir("training_functions.jl"))
include(srcdir("LSTMModels.jl"))
include(srcdir("LSTM_training_functions.jl"))
include(srcdir("LatentODEModels.jl"))
include(srcdir("LatentODE_training_functions.jl"))
include(srcdir("evaluating_functions.jl"))
include(srcdir("stats_and_plotting_functions.jl"))

# Export functions
export generate_dataset
export training_pipeline
export Cos4Exp
export start_up_workers
export LSTM_autoencoder, LatentODE
export TestingDataset_forecast
export get_scores, get_stats_and_plots_df, get_p_vals, plot_scores

end