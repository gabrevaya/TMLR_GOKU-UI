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
    :model_type => [GOKU_attention(), GOKU_basic()],    # base model  

    ## Latent Differential Equations
    :diffeq => Stoch_Hopf,
    :diffeq_args => Dict(:N => 3),                      # optional system and solver arguments
    :dt => 0.05f0,                                      # timestep for saving numerical solution

    ## Multiple shooting settings  
    :multiple_shooting => [true, false],                # multiple or single shooting training   
    :win_len => 10,                                     # window length for the multiple shooting
    :continuity_term => 2f0,                            # continuity regularization weight in the muliple shooting loss function        

    ## Data
    :training_samples => 75*[2^i for i in 0:6],         # number of samples used for training
    :val_samples => 200,                                # number of samples used for validation
    :data_path =>  datadir("sims", "data_Stoch_Hopf_3_samples=6000.h5"),
                                                        # path to the data file
    ## Training params
    :epochs => 20000,                                   # maximum number of epochs for training
    :batch_size => 64,                                  # minibatch size
    :seq_len => 46,                                     # approximate sequence length for training samples
                                                        # (it may be adjusted to make it compatible with multiple shooting)
    :optimizer => AdamW,                                # optimizer
    :lr => 0.005251,                                    # base learning rate for the schedule
    :decay_of_momentums => (0.9, 0.999),                # decay of momentums
    :ϵ => 1.0e-8,                                       # ϵ for ADAM
    :weight_decay => 1e-10,                             # weight decay for ADAMW

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
    :initial_wait_β => 0,
    :annealing_γ => 0.999999,
    :annealing_length => 0,

    ## Networks sizes and activation functions
    :hidden_dim_resnet => 200,
    :rnn_input_dim => 128,
    :rnn_output_dim => 64,
    :latent_dim_z₀ => 64,
    :latent_dim_θ => 128,
    :latent_to_diffeq_dim => 200,
    :general_activation => mish,
    :z₀_activation => identity,
    :θ_activation => σ,
    :output_activation => identity,
    :init => Flux.kaiming_uniform(gain = 1/sqrt(3)),

    ## Other
    :seed => collect(1:10),                             # random seed
    :cuda => false,                                     # GPU usage if available
    :verbose => true,
    :experiments_name => "GOKU_data_scaling",
    :name => "run1",
    :comments => "large param ranges in the model diffeq but small ranges in the data",
    :resume => false,
    :save_checkpoints => false,
    :save_output => true,
)

dicts = dict_list(general_args);

map(training_pipeline, dicts)