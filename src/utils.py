import torch
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
    
    # Load the configuration from config.json
    with open('src/config.json', 'r') as config_file:
        config = json.load(config_file)
    
    

    model_version = config["model_version"]

    if model_version == "v1":
        from models.v1.tokenizer import ChessTokenizer
        from models.v1.chess_dataset import ChessDataset
        # from models.v1.model import ChessGPTModel
        tok = ChessTokenizer()
        vocab_size = tok.vocabulary_size()
        config_gpt = GPT2Config(vocab_size=vocab_size, n_positions=config["seq_len"], n_ctx=config["seq_len"], n_embd=config["hidden_size"], n_layer=config["n_layer"], n_head=config["n_head"])
        model = GPT2LMHeadModel(config_gpt)
        chess_dataset = ChessDataset(tok, from_local=False, prefix="", max_seq_len=config["seq_len"])

    else:
        from tokenizer import ChessTokenizer
        tok = ChessTokenizer()
        vocab_size = tok.vocabulary_size()
        config_gpt = GPT2Config(vocab_size=vocab_size, n_positions=config["seq_len"], n_ctx=config["seq_len"], n_embd=config["hidden_size"], n_layer=config["n_layer"], n_head=config["n_head"])
        model = ChessGPTModel(config_gpt)
        chess_dataset = None

    
    
    model.to(gpu_device)

    num_of_parameters = sum(map(torch.numel, model.parameters()))
    if to_print: print(f"Number of parameters: {num_of_parameters}")

    return model, tok, config, gpu_device, chess_dataset