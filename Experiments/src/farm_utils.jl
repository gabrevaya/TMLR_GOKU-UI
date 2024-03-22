# Adapted from https://github.com/magerton/FARMTest.jl

"workers() but excluding master"
getworkers() = filter(i -> i != 1, workers())

function start_up_workers(ENV::Base.EnvDict; nprocs = Sys.CPU_THREADS)
    # send primitives to workers
    oldworkers = getworkers()
    @info "Removing workers $oldworkers"
    rmprocs(oldworkers)
    flush(stdout)
    
    if "SLURM_JOBID" in keys(ENV)
        num_cpus_to_request = parse(Int, ENV["SLURM_NTASKS"])
        @info "Requesting $(num_cpus_to_request) cpus from slurm"
        pids = addprocs(SlurmManager(); exeflags = "--project=$(Base.active_project())", topology=:master_worker)
    else
        cputhrds = Sys.CPU_THREADS
        cputhrds < nprocs && @warn "using nprocs = $cputhrds < $nprocs specified"
        pids = addprocs(min(nprocs, cputhrds); exeflags = "--project=$(Base.active_project())", topology=:master_worker)
    end
    @info "Workers added: $pids"
    return pids
end