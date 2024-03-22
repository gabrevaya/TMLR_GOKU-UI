####################################
# Default data generation arguments

function default_args_gen(diffeq::Hopf, tspan, dt, n_traj, transient_crop)

  N = length(diffeq.prob.u0)
  width, high = 28, 28
  mask_seed = Random.MersenneTwister(1)
  mask = Flux.Dense(N, width*high, init = Flux.kaiming_uniform(mask_seed))

  k = Int(length(diffeq.prob.u0)/2)

#   u₀_range = (-0.5f0, 0.5f0)
  u₀_range = (0.3f0, 0.4f0)
  u₀_range = [[u₀_range[1], u₀_range[2]] for i in 1:N]
  u₀_range = Flux.stack(u₀_range, dims=1)

#   a_range = (-0.2, 0.2)
  a_range = (0, 1) # we adjust it inside the equations
  a_range = [[a_range[1], a_range[2]] for i in 1:k]
  a_range = Flux.stack(a_range, dims=1)

#   f_range = (0.04, 0.07/)
  f_range = (0, 1) # we adjust it inside the equations
  f_range = [[f_range[1], f_range[2]] for i in 1:k]
  f_range = Flux.stack(f_range, dims=1)

#   C_range = (0, 0.2)
  C_range = (0, 1) # we adjust it inside the equations
  C_range = [[C_range[1], C_range[2]] for i in 1:k^2]
  C_range = Flux.stack(C_range, dims=1)

  p_range = vcat(a_range, f_range, C_range)
  p_range = Float32.(p_range)

  Dict(
    :tspan => tspan,                      # time span # crop the first 100 time steps to get rid of the transient
    :dt => dt,                            # timestep for ode solve
    :u₀_range => u₀_range,
    :p_range => p_range,                  # parameter value range
    :n_traj => n_traj,                    # Number of trajectories
    :high_dim_args => (mask = mask,
                        width = width,
                        high = high,
                        transient_crop = transient_crop),

    :seed => 1,                           # random seed
  )
end