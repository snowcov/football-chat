from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
import os

def prepare_and_tokenize_data(data_path, tokenizer_path, output_dir):
    """
    Function to load the dataset, apply the tokenizer, and save the tokenized dataset.
    
    Parameters:
    - data_path (str): Path to the JSONL dataset file.
    - tokenizer_path (str): Path to the pre-trained tokenizer.
    - output_dir (str): Directory to save the tokenized dataset.
    """
    # Load your JSONL dataset
    dataset = load_dataset("json", data_files={"train": data_path}, split="train")

    # Load the custom tokenizer you trained
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Tokenize the dataset
    def tokenize_function(examples):
        # Using encode_batch() to tokenize the batch of 'prompt' texts
        encoding = tokenizer.encode_batch(examples['prompt'])
        
        # Convert the encoding into a format compatible with Hugging Face's datasets (input_ids, attention_mask)
        return {
            'input_ids': [e.ids for e in encoding],
            'attention_mask': [[1] * len(e.ids) for e in encoding]  # Assume no padding yet; this can be adjusted
        }

    # Apply the tokenizer to the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the tokenized dataset
    tokenized_datasets.save_to_disk(os.path.join(output_dir, "tokenized_dataset"))

    print(f"Tokenized dataset saved to {os.path.join(output_dir, 'tokenized_dataset')}")

# Example usage:
if __name__ == "__main__":
    data_path = "./data/full_fantasy_stats_prompts.jsonl"  # Path to your JSONL data
    tokenizer_path = "fantasy_tokenizer/fantasy_tokenizer.json"      # Path to your custom tokenizer
    output_dir = "tokenized_data"                  # Directory to save the tokenized dataset
    
    prepare_and_tokenize_data(data_path, tokenizer_path, output_dir)
