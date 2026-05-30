# Comparison between Autoregressive LLMs and Diffusion LLMs

## Models
TODO

## Dataset
TODO

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
```
mkdir -p /scratch/<netid>/hf_cache

export HF_HOME=/scratch/<netid>/hf_cache
export TRANSFORMERS_CACHE=/scratch/<netid>/hf_cache

module load 2025
module load python

python -c "from transformers import AutoTokenizer, AutoConfig; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct'); AutoConfig.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct')"
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct')"

python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('inclusionAI/LLaDA2.1-mini'); AutoModelForCausalLM.from_pretrained('inclusionAI/LLaDA2.1-mini', trust_remote_code=True)"

python -c "from datasets import load_dataset; load_dataset('nuprl/CanItEdit')"
```

5. Use the following sbatch script to run the comparison:

```
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
