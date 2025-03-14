#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=job_%x_%j.log  # Unique standard output log
#SBATCH --error=job_%x_%j.err   # Unique error log
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000
#SBATCH --time=3-00:00:00
#SBATCH --job-name=MyTrainingJob  # Custom job name

# 🚀 1️⃣ Load the latest CUDA module (if available)
module purge  # Clears any previously loaded modules
module load cuda  # Loads the latest CUDA version available on the cluster

# 🚀 2️⃣ Automatically find CUDA installation path
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=0

# 🚀 3️⃣ Install CUDA and cuDNN (if not already installed)
if [ ! -d "$CUDA_HOME" ]; then
    echo "CUDA not found! Installing CUDA 11.8..."
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run -O cuda_installer.run
    chmod +x cuda_installer.run
    sudo ./cuda_installer.run --silent --toolkit
    export CUDA_HOME=/usr/local/cuda-11.8
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    echo "CUDA 11.8 installed!"
fi

# 🚀 4️⃣ Install cuDNN (if not installed)
if [ ! -d "/usr/local/cuda-11.8/lib64" ]; then
    echo "Installing cuDNN 8..."
    wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.4.1/cudnn-11.8-linux-x64-v8.4.1.50.tgz
    tar -xzvf cudnn-11.8-linux-x64-v8.4.1.50.tgz
    sudo cp -P cuda/include/cudnn*.h /usr/local/cuda-11.8/include/
    sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.8/lib64/
    sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*
    echo "cuDNN 8 installed!"
fi

# 🚀 5️⃣ Verify CUDA Installation
nvcc --version
nvidia-smi
echo "Using CUDA from: $CUDA_HOME"

# Set up output directory
export OUTPUT_DIR="/disk/scratch/$(whoami)/output/"
mkdir -p ${OUTPUT_DIR}

# Activate the virtual environment
source /home/$(whoami)/miniconda3/bin/activate mlpcw3
conda install -y fastai scikit-learn pandas "numpy==1.23.5" tqdm certifi

# Run Python script
LOG_FILE="${OUTPUT_DIR}/training_${SLURM_JOB_ID}.log"
stdbuf -oL -eL python -u main.py > $LOG
