#!/bin/bash

#SBATCH -A NAISS2024-22-205 -p alvis
#SBATCH --gpus-per-node=T4:1
#SBATCH --time=00:10:00
#SBATCH --mail-user=emastr@kth.se --mail-type=end


module purge
#virtualenv --system-site-packages deepmicro_test
module load virtualenv/20.23.1-GCCcore-12.3.0
module load matplotlib/3.7.2-gfbf-2023a 
module load SciPy-bundle/2023.07-gfbf-2023a 
module load Python/3.11.3-GCCcore-12.3.0
module load IPython/8.14.0-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

source deepmicro_test/bin/activate
pip install -e . build


#python ~/scripts/testing/pytorch_test.py
python ~/deep-micro-slip-model/notebooks/part30_paper_runs/part30_step2_training.py