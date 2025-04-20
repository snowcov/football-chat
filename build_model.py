import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

class FantasyFootballGPT(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=12, num_heads=12):
        super(FantasyFootballGPT, self).__init__()
        
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
        )
        self.transformer = GPT2LMHeadModel(config)
        
    def forward(self, input_ids, labels=None):
        return self.transformer(input_ids, labels=labels)
