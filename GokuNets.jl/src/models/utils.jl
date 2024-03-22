# to avoide having to import DiffEqFlux (which brings problems for Ensemble solve
# on SDEs):

"""
Get ranges that partition data of length `datasize` in groups of `groupsize` observations.
If the data isn't perfectly dividable by `groupsize`, the last group contains
the reminding observations. (originally from DiffEqFlux.jl)

```julia
group_ranges(datasize, groupsize)
```

Arguments:
- `datasize`: amount of data points to be partitioned
- `groupsize`: maximum amount of observations in each group

Example:
```julia-repl
julia> group_ranges(10, 5)
3-element Vector{UnitRange{Int64}}:
 1:5
 5:9
 9:10
```
"""
function group_ranges(datasize::Integer, groupsize::Integer)
    2 <= groupsize <= datasize || throw(
        DomainError(
            groupsize,
            "datasize must be positive and groupsize must to be within [2, datasize]",
        ),
    )
    return [i:min(datasize, i + groupsize - 1) for i in 1:groupsize-1:datasize-1]
end

# Helper function to calculate sequence lengths compatible with the multiple shooting splits
mult_shooting_seq_len(win_len, N) = (win_len * N) - (N-1)

function adjust_seq_len_for_multiple_shooting(seq_len, win_len)
    N₀ = seq_len ÷ win_len
    Ns = N₀ - 1 : N₀ + 1
    seq_len_candidates = mult_shooting_seq_len.(win_len, Ns)
    ind = findmin(abs.(seq_len_candidates .- seq_len))[2]
    return seq_len_candidates[ind]
end

# Function to split fe_out for multiple shooting
function split_for_multiple_shooting_with_buffer(fe_out, seq_len, win_len)
    ranges = group_ranges(seq_len, win_len)
    fe_out_splitted = Zygote.Buffer(fe_out, size(fe_out, 1), win_len, size(fe_out, 2)*length(ranges))
    for i in 1:size(fe_out, 2)
        for j in 1:length(ranges)
            fe_out_splitted[:, :, (i-1)*length(ranges) + j] = fe_out[:, i, ranges[j]]
        end
    end
    return copy(fe_out_splitted)
end

function split_for_multiple_shooting(x::Array{T, 3}, seq_len::Integer, win_len::Integer) where T
    ranges = group_ranges(seq_len, win_len)
    fe_out_splitted = splitter(x, ranges)
    fe_out_splitted = cat(fe_out_splitted..., dims=Val(3))
end

splitter(x::Array{T, 3}, ranges::Vector{UnitRange{Int64}}) where T = [stack([x[:, i, rg] for rg in ranges], dims=3) for i in 1:size(x, 2)]