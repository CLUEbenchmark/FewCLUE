#!/bin/bash
#SBATCH -o log.%j.job
#SBATCH -p normal 
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:1
module rm compiler/rocm/2.9
module load compiler/rocm/4.0.1

# load conda env
PYTHON_HOME=/public/home/cluebenchmark/anaconda3/envs/lm-bff
export PATH=$PYTHON_HOME/bin:$PATH
source ~/anaconda3/bin/activate
conda activate lm-bff

cd ~/lm-bff
srun python test.py
#srun python test.py -n=1
