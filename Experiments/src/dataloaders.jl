struct TrainingDataset
    path::String
    rng::MersenneTwister
    seq_len::Int64
    N_samples::Int64
end

abstract type GeneralValidationDataset end
struct ValidationDataset <: GeneralValidationDataset
    path::String
    rng::MersenneTwister
    seq_len::Int64
end

struct ValidationDataset_HighDimOnly <: GeneralValidationDataset
    path::String
    rng::MersenneTwister
    seq_len::Int64
end

struct TestingDataset <: GeneralValidationDataset
    path::String
    rng::MersenneTwister
    seq_len::Int64
end

struct TestingDataset_forecast <: GeneralValidationDataset
    path::String
    rng::MersenneTwister
    seq_len::Int64
    horizon::Int64
end

struct TestingDataset_fMRI_forecast <: GeneralValidationDataset
    path::String
    rng::MersenneTwister
    seq_len::Int64
    horizon::Int64
end

# For processing all sliding windows
struct ValidationDataset_HighDimOnly_temporal
    path::String
    rng::MersenneTwister
    seq_len::Int64
end

MLUtils.numobs(data::TrainingDataset) = data.N_samples

function MLUtils.numobs(data::GeneralValidationDataset)
    data_size = h5open(data.path, "r") do file
        obj = file["validation/high_dim_data"]
        size(obj)
    end
    # return data_size[end]
    # workaround for StackOverflowError in MacOS ARM Julia 1.8 for cat function on large arrays
    nobs = data_size[end]<900 ? data_size[end] : 900
    return nobs
end

function MLUtils.numobs(data::ValidationDataset_HighDimOnly_temporal)
    data_size = h5open(data.path, "r") do file
        obj = file["validation/high_dim_data"]
        size(obj)
    end
    return data_size[end]*(data_size[2] - data.seq_len + 1)
end

function MLUtils.getobs(data::ValidationDataset, idxs)
    seq_len = data.seq_len
    rng = data.rng
    batch_size = length(idxs)
    fid = h5open(data.path, "r")
    high_dim_data = fid["validation/high_dim_data"]
    latent_dim_data = fid["validation/latent_dim_data"]
    params_data = fid["validation/params_data"]
    high_dim_size = size(high_dim_data)
    latent_dim_size = size(latent_dim_data)
    params_size = size(params_data)

    # Randomly select sequence start
    starts = rand(rng, 1:high_dim_size[2] - seq_len + 1, batch_size)
    
    # Prepare data containers with shape (features, batch_size, time)
    high_dim_batch = Array{Float32}(undef, high_dim_size[1], batch_size, seq_len) 
    latent_dim_batch = Array{Float32}(undef, latent_dim_size[1], batch_size, seq_len) 
    params_batch = Array{Float32}(undef, params_size[1], batch_size)

    for j in 1:batch_size
        high_dim_batch[:, j, :] = high_dim_data[:, starts[j]:starts[j] + seq_len - 1, idxs[j]]
    end

    for j in 1:batch_size
        latent_dim_batch[:, j, :] = latent_dim_data[:, starts[j]:starts[j] + seq_len - 1, idxs[j]]
    end

    for j in 1:batch_size
        params_batch[:, j] = params_data[:, idxs[j]]
    end
    close(fid)

    return high_dim_batch, latent_dim_batch, params_batch
end


function MLUtils.getobs(data::ValidationDataset_HighDimOnly, idxs)
    seq_len = data.seq_len
    rng = data.rng
    batch_size = length(idxs)
    fid = h5open(data.path, "r")
    high_dim_data = fid["validation/high_dim_data"]
    high_dim_size = size(high_dim_data)

    # Randomly select sequence start
    starts = rand(rng, 1:high_dim_size[2] - seq_len + 1, batch_size)
    
    # Prepare data containers with shape (features, batch_size, time)
    high_dim_batch = Array{Float32}(undef, high_dim_size[1], batch_size, seq_len) 

    for j in 1:batch_size
        high_dim_batch[:, j, :] = high_dim_data[:, starts[j]:starts[j] + seq_len - 1, idxs[j]]
    end

    close(fid)

    return high_dim_batch, nothing, nothing
end


function MLUtils.getobs(data::ValidationDataset_HighDimOnly_temporal, idxs)
    seq_len = data.seq_len
    rng = data.rng
    batch_size = length(idxs)
    fid = h5open(data.path, "r")
    high_dim_data = fid["validation/high_dim_data"]
    high_dim_size = size(high_dim_data)

    # Randomly select sequence start
    starts = rand(rng, 1:high_dim_size[2] - seq_len + 1, batch_size)
    
    # Prepare data containers with shape (features, batch_size, time)
    high_dim_batch = Array{Float32}(undef, high_dim_size[1], batch_size, seq_len) 

    for j in 1:batch_size
        high_dim_batch[:, j, :] = high_dim_data[:, starts[j]:starts[j] + seq_len - 1, idxs[j]]
    end

    close(fid)

    return high_dim_batch, nothing, nothing
end

function MLUtils.getobs(data::TrainingDataset, idxs)
    seq_len = data.seq_len
    rng = data.rng
    batch_size = length(idxs)
    fid = h5open(data.path, "r")
    high_dim_data = fid["training/high_dim_data"]
    high_dim_size = size(high_dim_data)

    # Randomly select sequence start
    starts = rand(rng, 1:high_dim_size[2] - seq_len + 1, batch_size)

    # Prepare data container with shape (features, batch_size, time)
    high_dim_batch = Array{Float32}(undef, high_dim_size[1], batch_size, seq_len) 

    # Load required data
    for j in 1:batch_size
        high_dim_batch[:, j, :] = high_dim_data[:, starts[j]:starts[j] + seq_len - 1, idxs[j]]
    end
    close(fid)

    return high_dim_batch
end

function MLUtils.getobs(data::TestingDataset, idxs)
    seq_len = data.seq_len
    rng = data.rng
    batch_size = length(idxs)
    fid = h5open(data.path, "r")
    high_dim_data = fid["validation/high_dim_data"]
    latent_dim_data = fid["validation/latent_dim_data"]
    params_data = fid["validation/params_data"]
    high_dim_size = size(high_dim_data)
    latent_dim_size = size(latent_dim_data)
    params_size = size(params_data)

    # Prepare data containers with shape (features, batch_size, time)
    high_dim_batch = Array{Float32}(undef, high_dim_size[1], batch_size, seq_len) 
    latent_dim_batch = Array{Float32}(undef, latent_dim_size[1], batch_size, seq_len) 
    params_batch = Array{Float32}(undef, params_size[1], batch_size)

    for j in 1:batch_size
        high_dim_batch[:, j, :] = high_dim_data[:, 1:seq_len, idxs[j]]
    end

    for j in 1:batch_size
        latent_dim_batch[:, j, :] = latent_dim_data[:, 1:seq_len, idxs[j]]
    end

    for j in 1:batch_size
        params_batch[:, j] = params_data[:, idxs[j]]
    end
    close(fid)

    return high_dim_batch, latent_dim_batch, params_batch
end


function MLUtils.getobs(data::TestingDataset_forecast, idxs)
    seq_len = data.seq_len
    horizon = data.horizon
    rng = data.rng
    batch_size = length(idxs)
    fid = h5open(data.path, "r")
    high_dim_data = fid["validation/high_dim_data"]
    latent_dim_data = fid["validation/latent_dim_data"]
    params_data = fid["validation/params_data"]
    high_dim_size = size(high_dim_data)
    latent_dim_size = size(latent_dim_data)
    params_size = size(params_data)

    # Prepare reconstruction data containers with shape (features, batch_size, time)
    high_dim_batch = Array{Float32}(undef, high_dim_size[1], batch_size, seq_len) 
    latent_dim_batch = Array{Float32}(undef, latent_dim_size[1], batch_size, seq_len) 
    params_batch = Array{Float32}(undef, params_size[1], batch_size)

    # Prepare forecast data containers with shape (features, batch_size, time)
    high_dim_batch_target = Array{Float32}(undef, high_dim_size[1], batch_size, horizon) 
    latent_dim_batch_target = Array{Float32}(undef, latent_dim_size[1], batch_size, horizon) 

    for j in 1:batch_size
        high_dim_batch[:, j, :] = high_dim_data[:, 1:seq_len, idxs[j]]
        high_dim_batch_target[:, j, :] = high_dim_data[:, seq_len+1:seq_len+horizon, idxs[j]]
    end

    for j in 1:batch_size
        latent_dim_batch[:, j, :] = latent_dim_data[:, 1:seq_len, idxs[j]]
        latent_dim_batch_target[:, j, :] = latent_dim_data[:, seq_len+1:seq_len+horizon, idxs[j]]
    end

    for j in 1:batch_size
        params_batch[:, j] = params_data[:, idxs[j]]
    end
    close(fid)

    return (high_dim_batch, latent_dim_batch, params_batch), (high_dim_batch_target, latent_dim_batch_target)
end



function MLUtils.getobs(data::TestingDataset_fMRI_forecast, idxs)
    # thought for fMRI_training_data.h5
    seq_len = data.seq_len
    horizon = data.horizon
    batch_size = length(idxs)
    fid = h5open(data.path, "r")
    high_dim_data = fid["training/high_dim_data"]
    high_dim_size = size(high_dim_data)
    high_dim_data_target = fid["validation/high_dim_data"]
    high_dim_size_target = size(high_dim_data)

    # Prepare data containers with shape (features, batch_size, time)
    high_dim_batch = Array{Float32}(undef, high_dim_size[1], batch_size, seq_len) 
    high_dim_data_target_batch = Array{Float32}(undef, high_dim_size[1], batch_size, horizon) 

    for j in 1:batch_size
        high_dim_batch[:, j, :] = high_dim_data[:, end-seq_len+1:end, idxs[j]]
        high_dim_data_target_batch[:, j, :] = high_dim_data_target[:, 1:horizon, idxs[j]]
    end

    close(fid)

    return high_dim_batch, high_dim_data_target_batch
end
