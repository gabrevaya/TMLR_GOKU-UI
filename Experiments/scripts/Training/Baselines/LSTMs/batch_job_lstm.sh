#!/bin/bash -l

#-----------------------
# Job info
#-----------------------

#SBATCH --job-name=lstm
#SBATCH --mail-user=your@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=log/slurm-%x.%j.out
#SBATCH --error=log/slurm-%x.%j.err

#-----------------------
# Resource allocation
#-----------------------

#SBATCH --time=3-00:00:00     # in d-hh:mm:ss
#SBATCH --ntasks=20
#SBATCH --partition=long-cpu
#SBATCH --mem=140G

#-----------------------
# script
#-----------------------

echo ""
hostname

echo ""
echo "Starting job ${SLURM_JOB_ID} on partition ${SLURM_JOB_PARTITION}}"
echo ""

# print out environment variables
printenv | grep SLURM

echo ""
echo "Copying dataset to temp directory"
echo ""
# Copy the dataset on the compute node
dataset="data_Stoch_Hopf_3_samples=6000.h5"
path_to_codes="your_path_to_codes"
data_path="$path_to_codes/TMLR_GOKU-UI/Experiments/data/sims"
cp $data_path/$dataset $SLURM_TMPDIR

echo "Starting Julia"
# run training script
julia --project=$path_to_codes/TMLR_GOKU-UI/Experiments --optimize=3 --threads=1 $path_to_codes/TMLR_GOKU-UI/Experiments/scripts/Training/Baselines/LSTMs/train_lstm_cluster.jl "$SLURM_TMPDIR/$dataset"