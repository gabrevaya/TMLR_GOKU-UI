using DrWatson
@quickactivate "TMLR_experiments"
using GokuNets
using TMLR_experiments

# small range of parameters (used for generating Hopf data)
GokuNets.linear_scale_a(x) = x*0.4f0 - 0.2f0
GokuNets.linear_scale_ω(x) = π*(x*0.06f0 + 0.08f0)  # to set it between 0.04 and 0.07 Hz
GokuNets.linear_scale_C(x) = x*0.2f0

################################################################################################################
## Testing dataset generation
config = Dict(
    :diffeq => Stoch_Hopf,                      
    :diffeq_args => Dict(:N => 3),              # optional system and solver arguments
    :tspan => (0.0f0, 14.95f0),                 # time span 
    :dt => 0.05f0,                              # timestep for ode solve
    :n_traj => 1000,                            # Number of trajectories
    :transient_crop => 100,                     # cropping of initial time steps to get rid of the transient
    :val_samples => 1000,
    :data_path => datadir("sims"),
    :seed => 3,                                 # random seed
    :cuda => false,                             # GPU usage if available
    :verbose => true,
)

@time generate_dataset(config)

################################################################################################################
## Training dataset generation
config = Dict(
    :diffeq => Stoch_Hopf,                      
    :diffeq_args => Dict(:N => 3),              # optional system and solver arguments
    :tspan => (0.0f0, 34.95f0),                 # time span 
    :dt => 0.05f0,                              # timestep for ode solve
    :n_traj => 6000,                            # Number of trajectories
    :transient_crop => 100,                     # cropping of initial time steps to get rid of the transient
    :val_samples => 200,
    :data_path => datadir("sims"),
    :seed => 3,                                 # random seed
    :cuda => false,                             # GPU usage if available
    :verbose => true,
)

@time generate_dataset(config)