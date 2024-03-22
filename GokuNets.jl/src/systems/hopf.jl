################################################################################
## Problem Definition -- Normal form of a supercritical Hopf bifurcation
# Deco et al. (2017), https://www.nature.com/articles/s41598-017-03073-5
# Ipiña et al. (2019), https://arxiv.org/abs/1907.04412 & https://www.sciencedirect.com/science/article/pii/S1053811920303207

abstract type System end
abstract type Hopf <: System end

struct Stoch_Hopf{P,S,T,K} <: Hopf

    prob::P
    solver::S
    sensealg::T
    kwargs::K

    function Stoch_Hopf(; N = 1, solver = SOSRI(), sensealg = SciMLSensitivity.ForwardDiffSensitivity(), kwargs...)
        # Default parameters and initial conditions
        a = 0.2f0*randn(Float32, N)
        f = (fill(0.055f0, N) + 0.03f0*randn(Float32, N)) # f = ω/2π
        C = 0.2f0*rand(Float32, N^2)
        u₀ = rand(Float32, 2*N)
        p = [a; f; C]
        t = range(0.f0, step = 0.05f0, length = 50)
        tspan = (t[1], t[end])

        # Define differential equations
        function f!(du,u,p,t)

            N = length(u) ÷ 2
            a = linear_scale_a.(p[1:N])     
            ω = linear_scale_ω.(p[N+1:2*N])
            C_vec = linear_scale_C.(p[2*N+1:end])

            C = reshape(C_vec, (N, N))
            G = 0.1f0

            x = u[1:N]
            y = u[N+1:end]

            for j ∈ 1:N
                coupling_x = 0.f0
                coupling_y = 0.f0
                for i ∈ 1:N
                    coupling_x += C[i,j]*(x[i] - x[j])
                    coupling_y += C[i,j]*(y[i] - y[j])
                end

                du[j] = ((a[j] - x[j]^2 - y[j]^2) * x[j] - ω[j]*y[j] + G*coupling_x)*20
                du[j+N] = ((a[j] - x[j]^2 - y[j]^2) * y[j] + ω[j]*x[j] + G*coupling_y)*20
            end
        end

        function σ(du,u,p,t)
            du .= 0.02f0
        end

        W = WienerProcess(0.0,0.0,0.0)
        _prob = SDEProblem(f!, σ, u₀, tspan, p, noise = W)
        sys = modelingtoolkitize(_prob)
        SDEFunc = SDEFunction{true}(sys, tgrad=true, jac = true, sparse = false, simplify = false)
        prob_sde = SDEProblem{true}(SDEFunc, σ, u₀, tspan, p)

        P = typeof(prob_sde)
        S = typeof(solver)
        T = typeof(sensealg)
        K = typeof(kwargs)
        new{P,S,T,K}(prob_sde, solver, sensealg, kwargs)
    end
end

# small range of parameters (used for generating the data)
# linear_scale_a(x) = x*0.4f0 - 0.2f0
# linear_scale_ω(x) = π*(x*0.06f0 + 0.08f0)  # to fix it between 0.04 and 0.07 Hz
# linear_scale_C(x) = x*0.2f0

# larger range of parameters
linear_scale_a(x) = x*2f0 - 1f0
linear_scale_ω(x) = x
linear_scale_C(x) = x*0.4f0 - 0.2f0