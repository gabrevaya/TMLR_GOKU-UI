using DrWatson
@quickactivate "TMLR_experiments"

using TMLR_experiments
using DataFrames
using JLD2

using GokuNets
import GokuNets.linear_scale_a
import GokuNets.linear_scale_ω
import GokuNets.linear_scale_C

# larger range of parameters and positive connectivity
GokuNets.linear_scale_a(x) = x*2f0 - 1f0
GokuNets.linear_scale_ω(x) = x
GokuNets.linear_scale_C(x) = x*0.2f0

testing_data = datadir("sims", "data_Stoch_Hopf_3_samples=1000.h5")
forecast_horizon = 20

@info "Evaluations"
@info "GOKU-nets"
folder = "exp_pro/GOKU_data_scaling"
df = collect_results(datadir(folder))
df_goku = get_scores(df, testing_data, forecast_horizon)

@info "LSTMs"
folder = "exp_pro/LSTM_data_scaling"
df = collect_results(datadir(folder))
df_lstm = get_scores(df, testing_data, forecast_horizon)

@info "LatentODEs"
folder = "exp_pro/LatentODE_data_scaling"
df = collect_results(datadir(folder))
df_latent_ode = get_scores(df, testing_data, forecast_horizon)

df_scores = vcat(df_goku, df_lstm, df_latent_ode, cols=:union)

@save projectdir("results", "df_scores.jld2") df_scores
@info "Done"
