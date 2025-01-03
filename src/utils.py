import torch
from tokenizer import ChessTokenizer
from transformers import GPT2Config, GPT2LMHeadModel
import json
from model import ChessGPTModel

def setup(to_print=True):
    cpu_device = torch.device("cpu")

    # Check if MPS backend is available
    if torch.backends.mps.is_available():
        if to_print:
            print("Using MPS backend")
        gpu_device = torch.device("mps")
    else:
        if to_print:
            print("Using Cuda backend")
        gpu_device = torch.device("cuda")
    
    tok = ChessTokenizer()

    # Load the configuration from config.json
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    vocab_size = tok.vocabulary_size()

    config_gpt = GPT2Config(vocab_size=vocab_size, n_positions=config["seq_len"], n_ctx=config["seq_len"], n_embd=config["hidden_size"], n_layer=config["n_layer"], n_head=config["n_head"])
    # model = GPT2LMHeadModel(config_gpt)
    model = ChessGPTModel(config_gpt)
    model.to(gpu_device)

    num_of_parameters = sum(map(torch.numel, model.parameters()))
    if to_print:
        print(f"Number of parameters: {num_of_parameters}")

    return model, tok, config, gpu_device