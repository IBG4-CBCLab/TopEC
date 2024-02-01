#!/bin/bash -x

#SBATCH --account vsk33
#SBATCH --output /p/project/vsk33/vanderweg/TopEC/slurmlogs/train_%j.job
#SBATCH --error /p/project/vsk33/vanderweg/TopEC/slurmlogs/train_%j.job
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 12
#SBATCH --time 01:00:00
#SBATCH --nodes 4
#SBATCH --partition dc-gpu
#SBATCH --gres gpu:4
#SBATCH -J train


echo 'SLURM JOB ID'
echo $SLURM_JOB_ID
echo 'Loading bashrc'
source $HOME/.bashrc
echo 'Starting training'

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

source /p/project/vsk33/vanderweg/TopEC/topec_venv/bin/activate

#run
srun python -u train.py --config-name train experiment=$1
