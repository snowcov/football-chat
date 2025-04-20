import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from datasets import load_from_disk
from tokenizers import Tokenizer  # Import Tokenizer from the tokenizers library
from build_model import FantasyFootballGPT

def collate_fn(batch):
    input_ids = [torch.tensor(example["input_ids"]) for example in batch]

    # Pad the sequences to the maximum length in the batch
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = input_ids.clone()
    
    return {"input_ids": input_ids, "labels": labels}


# Load the tokenized dataset from disk (this should be the output from data_prep.py)
tokenized_datasets = load_from_disk("tokenized_data/tokenized_dataset")  # Change path if needed

# Load the tokenizer used during training
tokenizer = Tokenizer.from_file("fantasy_tokenizer/fantasy_tokenizer.json")

# Ensure that the special tokens are defined in the tokenizer
special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
for token in special_tokens:
    if token not in tokenizer.get_vocab():
        tokenizer.add_tokens([token])

print(f"Tokenizer vocab size: {len(tokenizer.get_vocab())}")

# Initialize the model with the correct vocabulary size
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FantasyFootballGPT(vocab_size=len(tokenizer.get_vocab())).to(device)

# Resize token embeddings to match the tokenizer's vocabulary size
model.transformer.resize_token_embeddings(len(tokenizer.get_vocab()))

# Set up DataLoader for batching
train_loader = DataLoader(tokenized_datasets, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Define the optimizer (AdamW) and the loss function (CrossEntropyLoss)
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss()

# Training Loop
epochs = 3  # You can change this as needed
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
    for batch in loop:
        # Move the batch to the correct device (GPU or CPU)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()  # Reset gradients

        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        # Backpropagation
        loss.backward()
        
        # Gradient update
        optimizer.step()

        # Update progress bar with loss value
        loop.set_postfix(loss=loss.item())
    
    # Save model checkpoint at the end of each epoch
    model_checkpoint = f"fantasy_football_model_epoch_{epoch + 1}.pt"
    torch.save(model.state_dict(), model_checkpoint)
    print(f"✅ Model saved after epoch {epoch + 1} as {model_checkpoint}")

print(f"Tokenizer vocab size during training: {len(tokenizer.get_vocab())}")
