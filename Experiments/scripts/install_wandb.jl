using Conda

Conda.add("wandb")
ENV["PYTHON"]=""

using Pkg
Pkg.build("PyCall")

# restart julia
# now you can add, build and use Wandb.jl
