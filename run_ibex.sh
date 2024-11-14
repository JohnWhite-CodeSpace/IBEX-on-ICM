#!/bin/bash
#SBATCH --job-name=ibex_data_analysis
#SBATCH --output=ibex_data_analysis_%j.log
#SBATCH --error=ibex_data_analysis_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=plgrid
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@domain.com

module load plgrid/tools/python/3.8

python3 -m venv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

pip install torch numpy pyyaml

echo "Running tensor creation and analysis program..."
python /path/to/your/script.py

deactivate
echo "Job finished."