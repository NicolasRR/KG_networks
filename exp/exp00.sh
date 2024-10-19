#!/bin/bash

cd /scratch/reategui/KG_networks
source activate eureka-slm
export LD_LIBRARY_PATH=/scratch/reategui/conda/envs/eureka-slm/lib/

python main.py wandb=true
