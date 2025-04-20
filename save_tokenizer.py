from transformers import GPT2TokenizerFast

# Point to the correct file path inside the directory
tokenizer = GPT2TokenizerFast(tokenizer_file="fantasy_tokenizer/fantasy_tokenizer.json")

# Save the full tokenizer setup (with config files) to the same folder
tokenizer.save_pretrained("fantasy_tokenizer")

print("✅ Tokenizer saved and ready in 'fantasy_tokenizer/'")
