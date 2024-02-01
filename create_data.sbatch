#!/bin/bash -x

#SBATCH --account vsk33
#SBATCH --output /p/project/vsk33/vanderweg/topec/slurmlogs/createdata_%j.job
#SBATCH --error /p/project/vsk33/vanderweg/topec/slurmlogs/createdata_%j.job
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 1
#SBATCH --time 12:00:00
#SBATCH --nodes 1
#SBATCH --partition dc-gpu
#SBATCH --gres gpu:4
#SBATCH -J createdata

echo 'SLURM JOB ID'
echo $SLURM_JOB_ID
echo 'Loading bashrc'
source $HOME/.bashrc
echo 'Starting training'

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
source /home/${USER}/.bashrc
source /p/project/vsk33/vanderweg/TopEC/topec_venv/bin/activate

#run
python create_h5dataset.py