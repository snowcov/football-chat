from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import json

# Initialize a Byte-Level BPE tokenizer
tokenizer = Tokenizer(models.BPE())

# Set up pre-tokenization (to handle punctuation, etc.)
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# Define the special tokens
special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]

# Set up the trainer
trainer = trainers.BpeTrainer(vocab_size=5000, min_frequency=2, special_tokens=special_tokens)

# Provide your text data (train on your fantasy football dataset)
files = ["./data/full_fantasy_stats_prompts.jsonl", "./data/fantasy_rank_prompts.jsonl"]
tokenizer.train(files, trainer)

# Save the trained tokenizer (this will save the tokenizer to the directory)
tokenizer.save("fantasy_tokenizer/fantasy_tokenizer.json")

# Manually save the tokenizer configuration for Hugging Face compatibility
config = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>"
}
with open("fantasy_tokenizer/tokenizer_config.json", "w") as f:
    json.dump(config, f)

# Print the vocabulary size
print(f"Tokenizer vocab size: {len(tokenizer.get_vocab())}")

print("Tokenizer training complete and saved successfully!")
