import torch.nn as nn
from transformers import GPT2Config, GPT2Model

# mps_device = torch.device("mps")
# cpu_device = torch.device("cpu")

class ChessGPTModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super(ChessGPTModel, self).__init__()
        # self.config = GPT2Config(vocab_size=vocab_size, n_positions=seq_len, n_embd=hidden_size, n_layer=n_layer, n_head=n_head)
        self.config = config
        self.transformer = GPT2Model(self.config)
        self.linear_head = nn.Linear(config.n_embd, config.vocab_size)


    
    def forward(self, input_ids, attention_mask):

        transformer_output = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = transformer_output.last_hidden_state
        logits = self.linear_head(hidden_states)
        return logits