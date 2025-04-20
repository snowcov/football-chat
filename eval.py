from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from build_model import FantasyFootballGPT
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer using `tokenizers` (raw .json) to match training
tokenizer_raw = Tokenizer.from_file("fantasy_tokenizer/fantasy_tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_raw)
print("Tokenizer vocab size:", len(tokenizer))

# Initialize and load model
model = FantasyFootballGPT(vocab_size=len(tokenizer)).to(device)
model.load_state_dict(torch.load("fantasy_football_model_epoch_3.pt", map_location=device))
model.eval()

# Greedy decoding function
def generate_text(model, tokenizer, input_text, max_length=100):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    generated = input_ids

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated = torch.cat((generated, next_token_id), dim=1)
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Test generation
input_text = "Who is the top QB?"
print("Generated Response:", generate_text(model, tokenizer, input_text))
