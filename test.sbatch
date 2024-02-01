#!/bin/bash -x

#SBATCH --account vsk33
#SBATCH --output /p/project/vsk33/vanderweg/TopEC/slurmlogs/test_%j.job
#SBATCH --error /p/project/vsk33/vanderweg/TopEC/slurmlogs/test_%j.job
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 12
#SBATCH --time 01:00:00
#SBATCH --nodes 1
#SBATCH --partition dc-gpu
#SBATCH --gres gpu:4
#SBATCH -J test_TopEC

echo 'SLURM JOB ID'
echo $SLURM_JOB_ID
echo 'Loading bashrc'
source $HOME/.bashrc
echo 'Starting training'

export CUDA_VISIBLE_DEVICES=0,1,2,3

source /p/project/vsk33/vanderweg/TopEC/topec_venv/bin/activate

srun python -u test.py --config-name test experiment=$1