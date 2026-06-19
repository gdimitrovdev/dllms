# Comparison between Autoregressive LLMs and Diffusion LLMs

## Models
**Autoregressive Models:**
- Qwen2.5-Coder-7B-Instruct (`Qwen/Qwen2.5-Coder-7B-Instruct`)
- DeepSeek-Coder-6.7B-Instruct (`deepseek-ai/deepseek-coder-6.7b-instruct`)
- Qwen3-8B (`Qwen/Qwen3-8B`)

**Diffusion Models:**
- Dream-Coder-v0-Instruct-7B (`Dream-org/Dream-Coder-v0-Instruct-7B`)
- LLaDA-8B-Instruct (`GSAI-ML/LLaDA-8B-Instruct`)
- DiffuCoder-7B-cpGRPO (`apple/DiffuCoder-7B-cpGRPO`)

## Dataset
The evaluation is performed using the `nuprl/CanItEdit` dataset from Hugging Face.

## How to run locally
1. Create and activate a python virtual environment:
```
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the comparison:
```
python main.py
```

## How to run on DelftBlue
1. SSH into a login node:
```
ssh <netid>@login.delftblue.tudelft.nl
```

2. Download and run the Miniforge installer:
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

3. Create a conda virtual environment and setup packages:
```
conda create -n dllms python=3.11 -y
conda activate dllms
pip install -r <dllms_path>/requirements.txt
```

4. Download the models and dataset for offline access from your /scratch folder:
```bash
mkdir -p /scratch/<netid>/hf_cache

export HF_HOME=/scratch/<netid>/hf_cache
export TRANSFORMERS_CACHE=/scratch/<netid>/hf_cache

module load 2025
module load python

python download_models.py
```

5. Use the following sbatch script to run the comparison:

```bash
#!/bin/bash
#
#SBATCH --job-name="dllms"
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --account=<your-account-name>

source ~/miniforge3/etc/profile.d/conda.sh
conda activate dllms

cd /home/<netid>/dllms

export HF_HOME=/scratch/<netid>/hf_cache
export TRANSFORMERS_CACHE=/scratch/<netid>/hf_cache
export TOKENIZERS_PARALLELISM=false

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

srun python main.py
```

## Running Models Separately (Transformers Versions)
Because of compatibility issues, different `transformers` versions are required for the different model families:
- **Autoregressive Models** run smoothly with the `transformers` version in `requirements.txt` (which includes a custom shim in the pipeline).
- **Diffusion Models** require an older `transformers` version (e.g., `transformers==4.38.2`) due to custom generation code and masking utilities.

You will need to make separate runs for AR models and Diffusion models by commenting out the respective groups in `pipeline.py` and running them in separate conda environments with the appropriate `transformers` versions installed.

## Caching Mechanisms
This project employs two main caching mechanisms:
1. **Hugging Face Cache**: Model weights and the `nuprl/CanItEdit` dataset are cached in `HF_HOME`. Run the `download_models.py` script to populate this cache locally before running jobs in an offline cluster environment.
2. **Results Cache (`cache_artifacts/`)**: The outputs from each model (prefix and suffix generations) and their evaluated metrics are saved as JSON files in the `cache_artifacts/` directory. If a run is interrupted or if you restart the pipeline, it will automatically load cached generation results, bypassing the expensive model inference steps.
