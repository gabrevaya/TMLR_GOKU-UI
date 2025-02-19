
using DrWatson
@quickactivate "TMLR_experiments"
using GokuNets
using TMLR_experiments
using Flux
GokuNets.linear_scale_C(x) = x*0.2f0

################################################################################################################
## Arguments for the train function

general_args = Dict(
    ## Global model
    :model_type => LSTM_autoencoder(),                  # base model

    ## Data                                        
    :training_samples => 75*[2^i for i in 0:6],         # number of samples used for training
    :val_samples => 200,                                # number of samples used for validation
    :data_path => datadir("sims", "data_Stoch_Hopf_3_samples=6000.h5"),
                                                        # path to the data file
    ## Training params
    :epochs => 20000,                                   # maximum number of epochs for training
    :batch_size => 64,                                  # minibatch size
    :optimizer => ADAMW,                                # optimizer
    :lr => 0.005251,                                    # base learning rate for the schedule
    :decay_of_momentums => (0.9, 0.999),                # decay of momentums
    :ϵ => 1.0e-8,                                       # ϵ for ADAM
    :weight_decay => 1e-10,                             # weight decay for ADAMW
    :seq_len => 46,                                     # sequence length for training samples
    :dt => 0.05,

    # Learning rate scheduler
    :logging_and_scheduling_period => 9,                # interval for logging and lr schedule, measured in batch count
    :lr_scheduler => Cos4Exp,                           # main learning rate scheduler: Cos4Exp or Exp
    :min_lr => 0.00001,                                 # lower bound for the learning rate
    :warmup => 20*9,                                    # batches of initial linear growth of the lr until reaching the base lr
    :patience_lr => 50,                                 # patience (measured in logging_and_scheduling_periods) until reducing the lr for the first time
    :patience_lr2 => 2,                                 # patience (measured in logging_and_scheduling_periods) until reducing the lr for the subsequent times
    :threshold => 0.01,                                 # dynamic threshold for measuring the new optimum. See https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    :period => 50,                                      # period for the cosine/triangle lr schedules
    :γ => 0.995,                                        # exponential decay for the CosExp or Exp schedules
    :patience => 350,                                   # epochs (measured in logging_and_scheduling_periods) until early stopping of the training

    ## Variational settings
    :variational => false,                              # deterministic or variational training
    :initial_wait_β => 60,
    :annealing_γ => 0.998,
    :annealing_length => 5000*9,

    ## Networks sizes and activation functions
    :hidden_dim_resnet => 200,
    :rnn_input_dim => 128,
    :rnn_output_dim => 64,
    :z_dim => 42,
    :latent_dim_z₀ => 64,
    :latent_to_diffeq_dim => 200,
    :z₀_activation => mish,
    :general_activation => mish,
    :output_activation => identity,
    :init => Flux.kaiming_uniform(gain = 1/sqrt(3)),

    ## Other
    :seed => collect(1:10),                             # random seed
    :cuda => false,                                     # GPU usage if available
    :verbose => true,
    :experiments_name => "LSTM_data_scaling",
    :name => "run1",
    :resume => false,
    :save_checkpoints => false,
    :save_output => true,
)

dicts = dict_list(general_args)

map(training_pipeline, dicts)
