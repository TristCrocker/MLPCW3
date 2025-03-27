#!/bin/sh
#SBATCH -N 1       # Number of nodes
#SBATCH -n 1       # Number of tasks
#SBATCH --output=job_%j.log  # Unique standard output log file
#SBATCH --error=job_%j.err   # Unique error log file
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=16000  # Memory in MB
#SBATCH --time=3-00:00:00

# Set up CUDA
export CUDA_HOME=/opt/cuda-12.1.1/
export CUDNN_HOME=/usr/local/cuda
export STUDENT_ID=$(whoami)
export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH
export CPATH=${CUDNN_HOME}/include:$CPATH
export PATH=${CUDA_HOME}/bin:${PATH}
export PYTHON_PATH=$PATH
export CUDA_VISIBLE_DEVICES=0

# Set up scratch storage
export TMPDIR=/disk/scratch/${STUDENT_ID}
export TMP=/disk/scratch/${STUDENT_ID}
mkdir -p ${TMP}


# Ensure datasets exist in scratch storage
export DATASET_DIR=${TMP}/datasets/
mkdir -p ${DATASET_DIR}

if [ ! -d "${DATASET_DIR}/train_v2" ]; then
    echo "Copying dataset to scratch space..."
    cp -ru /home/${STUDENT_ID}/MLPCW3/data/* ${DATASET_DIR}/
fi
echo "Dataset is now in ${DATASET_DIR}"
echo $CUDA_VISIBLE_DEVICES

# Set up output directory
export OUTPUT_DIR=${TMP}/output/
mkdir -p ${OUTPUT_DIR}

if [ -f "/home/${STUDENT_ID}/MLPCW3/results/detr_model_bbox_pretrained_10.pth" ]; then
    cp /home/${STUDENT_ID}/MLPCW3/results/detr_model_bbox_pretrained_10.pth ${OUTPUT_DIR}/
    echo "Model copied to scratch space."
else
    echo "Warning: Model file not found in home directory. Training a new model."
fi

cp -ru /home/${STUDENT_ID}/MLPCW3/data/pretrained_model ${DATASET_DIR}/

# conda config --add channels defaults
# conda config --add channels conda-forge
conda config --set channel_priority flexible
# conda config --set channel_priority strict
# conda config --set solver classic
# Activate the virtual environment
source /home/${STUDENT_ID}/miniconda3/bin/activate mlpcw3
conda install -y fastai scikit-learn pandas timm transformers "numpy==1.23.5" tqdm certifi pytorch torchvision torchaudio -c pytorch --verbose 

# Run the Python script
cd /home/${STUDENT_ID}/MLPCW3/
stdbuf -oL -eL python -u main.py

# Save results to home directory
RESULTS_DIR=/home/${STUDENT_ID}/MLPCW3/results/
mkdir -p ${RESULTS_DIR}
cp -r ${OUTPUT_DIR}/* ${RESULTS_DIR}/

echo "Results saved to ${RESULTS_DIR}"
