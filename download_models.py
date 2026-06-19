import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from datasets import load_dataset

from pipeline import MODEL_REGISTRY, _model_load_class

def download_models_and_dataset():
    print("Downloading nuprl/CanItEdit dataset...")
    try:
        load_dataset("nuprl/CanItEdit")
        print("Successfully downloaded dataset.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

    for model_key, spec in MODEL_REGISTRY.items():
        model_id = spec["model_id"]
        print(f"\nDownloading tokenizer and model for {model_id}...")
        
        kwargs = {}
        if spec.get("trust_remote_code"):
            kwargs["trust_remote_code"] = True
            
        try:
            print(f"  -> Downloading tokenizer...")
            AutoTokenizer.from_pretrained(model_id, **kwargs)
            
            print(f"  -> Downloading model weights...")
            load_class = _model_load_class(spec)
            load_class.from_pretrained(model_id, **kwargs)
            
            print(f"Successfully downloaded {model_id}.")
        except Exception as e:
            print(f"Error downloading {model_id}: {e}")

if __name__ == "__main__":
    download_models_and_dataset()
