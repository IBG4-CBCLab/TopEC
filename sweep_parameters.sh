#!/bin/bash

declare -a StringArray=('experiment_01' 'experiment_02' 'experiment_03' 'experiment_04')

for model in ${StringArray[@]}; do
    for size in 16 20 24 28 32; do
        sbatch train.sbatch $model $size
        sleep 1
    done
done
