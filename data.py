from datasets import load_dataset

def load_canitedit_sample(split_name="test", index=0):
    """
    Loads a specific sample from the nuprl/CanItEdit dataset.
    """
    print("Loading CanItEdit dataset from Hugging Face...")
    dataset = load_dataset("nuprl/CanItEdit")

    if split_name not in dataset:
        split_name_fallback = list(dataset.keys())[0]
        print(f"Split {split_name} not found, using {split_name_fallback} instead.")
        split_name = split_name_fallback
        
    sample = dataset[split_name][index]
    before_code = sample["before"]
    instruction = sample["instruction_descriptive"]
    
    # TODO: Replace with automated sentence-splitting logic later
    mock_constraints = [
        "Modify the function to handle empty arrays.",
        "Ensure the time complexity remains O(n)."
    ]
    
    return before_code, instruction, mock_constraints

def load_all_canitedit_samples(split_name="test", limit=100):
    """
    Loads all samples up to limit from the nuprl/CanItEdit dataset.
    """
    print("Loading CanItEdit dataset from Hugging Face...")
    dataset = load_dataset("nuprl/CanItEdit")

    if split_name not in dataset:
        split_name_fallback = list(dataset.keys())[0]
        print(f"Split {split_name} not found, using {split_name_fallback} instead.")
        split_name = split_name_fallback
        
    split_dataset = dataset[split_name]
    num_samples = len(split_dataset)
    print(f"Dataset has {num_samples} samples in split '{split_name}'.")
    
    samples = []
    end_index = min(limit, num_samples) if limit else num_samples
    for index in range(end_index):
        sample = split_dataset[index]
        before_code = sample["before"]
        instruction = sample["instruction_descriptive"]
        
        # Keep the mock constraints as in original code
        mock_constraints = [
            "Modify the function to handle empty arrays.",
            "Ensure the time complexity remains O(n)."
        ]
        samples.append({
            "index": index,
            "before": before_code,
            "instruction": instruction,
            "constraints": mock_constraints
        })
    return samples

