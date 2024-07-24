#!/bin/bash

#SBATCH --job-name=train_model        # Job name
#SBATCH --output=train_expanded_text.out   # Output file name (%j expands to jobID)
#SBATCH --error=train_expanded_text.err    # Error file name (%j expands to jobID)
#SBATCH --time=50:00:00             # Maximum run time (3 days)
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=20            # Number of CPU cores per task
#SBATCH --gres=gpu:1                  # Number of GPUs
#SBATCH --partition=gpu               # Partition to submit to
#SBATCH --qos=gpu                     # QOS to submit to


# Load any modules that your application needs (if any)
module load python/3.8

# Activate the virtual environment
source /scratch/izar/mmorvan/EnergyEfficiencyPrediction/text/venv/bin/activate

# Navigate to the directory containing train.py
#cd /path/to/your/cloned/repository/model/models
cd /scratch/izar/mmorvan/EnergyEfficiencyPrediction/text

# Run the train.py script
python train_bert_all_features_24.py

# Deactivate the virtual environment
deactivate
