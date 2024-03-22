function generate_dataset(config)
      @unpack verbose, diffeq, diffeq_args = config
      @unpack val_samples, tspan, dt, n_traj, transient_crop = config

      verbose && @info "Building differential equation model"
      diffeq = diffeq(;diffeq_args...)   
  
      verbose && @info "Generating data"
      args_gen = default_args_gen(diffeq, tspan, dt, n_traj, transient_crop)
      high_dim_data, latent_data, params_data = generate_data(diffeq, args_gen, verbose)
  
      file_path = get_file_path(config, n_traj)
  
      # Make sure the element types are Float32
      @assert eltype(high_dim_data) == Float32 "The element type of high_dim_data is not Float32"
      @assert eltype(latent_data) == Float32 "The element type of latent_data is not Float32"
      @assert eltype(params_data) == Float32 "The element type of params_data is not Float32"
  
      # Split into train and validation sets
      high_dim_data_train = @view high_dim_data[:, :, 1:end-val_samples]
      high_dim_data_val = @view high_dim_data[:, :, end-val_samples+1:end]
      latent_dim_data_val = @view latent_data[:, :, end-val_samples+1:end]
      params_data_val = @view params_data[:, end-val_samples+1:end]
  
      @info "Saving data to $file_path"
      h5open(file_path, "w") do file
          write(file, "training/high_dim_data", high_dim_data_train)
          write(file, "validation/high_dim_data", high_dim_data_val)
          write(file, "validation/latent_dim_data", latent_dim_data_val)
          write(file, "validation/params_data", params_data_val)
      end
  
      verbose && @info "Done"
  
      return nothing
  end

function generate_data(diffeq::GokuNets.System, args_gen::Dict, verbose)
      @unpack tspan, dt, u₀_range, p_range, n_traj, high_dim_args, seed = args_gen

      if verbose 
            diffeq_type = diffeq |> typeof |> nameof
            timesteps = tspan[1]:dt:tspan[2] |> collect |> length
            timesteps -= high_dim_args.transient_crop
            diffeq_dim = length(diffeq.prob.u0)
            @info "  -----------------------------------"
            @info "  System: $diffeq_type"
            @info "  Latent dimensions: $diffeq_dim"
            @info "  Number of samples: $n_traj"
            @info "  Time steps: $timesteps"
            @info "  -----------------------------------"

      end
      Random.seed!(seed)

      ## Problem definition
      prob = remake(diffeq.prob, tspan = tspan)

      ## Ensemble functions definition
      prob_func(prob,i,repeat) = remake(prob, u0 = u0s[i], p = ps[i])

      # Sample initial condition and parameters from a uniform distribution
      ps = [Float32.(rand_uniform(p_range, length(prob.p))) for i in 1:n_traj]
      u0s = [Float32.(rand_uniform(u₀_range, length(prob.u0))) for i in 1:n_traj]

      # Build and solve EnsembleProblem data
      ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)

      verbose && @info "  Solving differential equations"
      sols = solve(ensemble_prob, diffeq.solver, saveat = dt, trajectories = n_traj)

      # Apply some transformation if needed (eg. for Kuramoto)
      sols = transform_after_diffeq(sols, diffeq)

      # Store the trajectories variables
      latent_data = sols[:, high_dim_args.transient_crop+1:end, :]

      verbose && @info "  Generating high dimensional data"

      # Create high dimensional data
      high_dim_data = create_high_dim_data(sols, high_dim_args, diffeq)

      ps = Flux.stack(ps, dims=2)

      return high_dim_data, latent_data, ps
end

## utility functions
rand_uniform(range::Tuple, size) = rand(Uniform(range...), (size,1))
rand_uniform(range::Array) = [rand(Uniform(r[1], r[2])) for r in eachrow(range)]
rand_uniform(range::Array, size) = rand_uniform(range)
transform_after_diffeq(x, diffeq) = identity(x)

apply_mask(x, mask, w, h) = [reshape(mask(col), w, h) for col in eachcol(x)] 

function create_high_dim_data(sols, high_dim_args, diffeq::GokuNets.System)
      mask, w, h, transient_crop = high_dim_args
      high_dim_data = mask(sols[:, transient_crop+1:end, :])
      return high_dim_data
end