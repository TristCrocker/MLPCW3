#!/bin/sh
#SBATCH -N 1       # Number of nodes
#SBATCH -n 1       # Number of tasks
#SBATCH --output=job_output.log  # Standard output log file
#SBATCH --error=job_error.log    # Standard error log file
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # Memory in MB
#SBATCH --time=3-00:00:00

# Load the latest CUDA version dynamically
module purge  # Clears any previously loaded modules
module load cuda  # Loads the highest available CUDA version

# Verify CUDA version
nvcc --version
nvidia-smi

# Set CUDA environment variables dynamically
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))  # Auto-detect CUDA path
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=0  # Use the assigned GPU

# Debugging: Print CUDA location
echo "Using CUDA from: $CUDA_HOME"

# Set up scratch storage
export STUDENT_ID=$(whoami)
export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/
mkdir -p ${TMP}

# Ensure datasets exist in scratch storage
export DATASET_DIR=${TMP}/datasets/
mkdir -p ${DATASET_DIR}

if [ ! -d "${DATASET_DIR}/train_v2" ]; then
    echo "Copying dataset to scratch space..."
    cp -r /home/${STUDENT_ID}/MLPCW3/data/* ${DATASET_DIR}/
fi
echo "Dataset is now in ${DATASET_DIR}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Set up output directory
export OUTPUT_DIR=${TMP}/output/
mkdir -p ${OUTPUT_DIR}

# Activate the virtual environment
source /home/${STUDENT_ID}/miniconda3/bin/activate mlpcw3
conda install -y fastai scikit-learn pandas "numpy==1.23.5" tqdm certifi

# Run the Python script
cd /home/${STUDENT_ID}/MLPCW3/
stdbuf -oL -eL python -u main.py

# Save results to home directory
RESULTS_DIR=/home/${STUDENT_ID}/MLPCW3/results/
mkdir -p ${RESULTS_DIR}
cp -r ${OUTPUT_DIR}/* ${RESULTS_DIR}/

echo "Results saved to ${RESULTS_DIR}"
