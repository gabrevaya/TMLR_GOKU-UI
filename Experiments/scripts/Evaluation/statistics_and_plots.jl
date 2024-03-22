using DrWatson
@quickactivate "TMLR_experiments"

using TMLR_experiments
using Random
using DataFrames
using JLD2
using MLUtils
using Plots

testing_data = datadir("sims", "data_Stoch_Hopf_3_samples=1000.h5")
rng = Random.MersenneTwister(3)
forecast_horizon = 20
seq_len = 46
d_test = TestingDataset_forecast(testing_data, rng, seq_len, forecast_horizon)
dataloader = MLUtils.DataLoader(d_test, batchsize=numobs(d_test), buffer=false, parallel=false,
                                        partial=false, rng=rng, shuffle=false)
                                        
(x, z, θ), (x_future, z_future) = first(dataloader)

@load projectdir("results", "df_scores.jld2") df_scores

@info "Reconstruction task"
rename!(df_scores, [:score_rec => "score"])
df_stats, df_plot = get_stats_and_plots_df(df_scores, x)

# Statistical tests
get_p_vals(df_stats)

# Plotting
plt = plot_scores(df_plot)
Plots.savefig(projectdir("results", "data_scaling_reconstruction.pdf"))

@info "Forecast task"
df_scores = df_scores[:, Not(:score)]
rename!(df_scores, [:score_for => "score"])
df_stats, df_plot = get_stats_and_plots_df(df_scores, x_future)

# Statistical tests
get_p_vals(df_stats)

# Plotting
plt = plot_scores(df_plot, ylabel = "Forecast NRMSE")
Plots.savefig(projectdir("results", "data_scaling_forecast.pdf"))