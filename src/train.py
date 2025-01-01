import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from torch.utils.data import DataLoader
import torch.optim as optim
from tokenizer import ChessTokenizer

import gc
from model import ChessGPTModel
from chessdata import ChessDataset

import json
from metrics import *


def setup():
    cpu_device = torch.device("cpu")

    # Check if MPS backend is available
    if torch.backends.mps.is_available():
        print("Using MPS backend")
        gpu_device = torch.device("mps")
    else:
        print("Using Cuda backend")
        gpu_device = torch.device("cuda")
    
    tok = ChessTokenizer()

    # Load the configuration from config.json
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    vocab_size = tok.vocabulary_size()

    config_gpt = GPT2Config(vocab_size=vocab_size, n_positions=config["seq_len"], n_ctx=config["seq_len"], n_embd=config["hidden_size"], n_layer=config["n_layer"], n_head=config["n_head"])
    model = GPT2LMHeadModel(config_gpt)
    model.to(gpu_device)

    num_of_parameters = sum(map(torch.numel, model.parameters()))
    print(f"Number of parameters: {num_of_parameters}")

    return model, tok, config, gpu_device





def main():
    model, tok, config, gpu_device = setup()

    batch_size = 16
    learning_rate = 1e-3
    num_epochs = 100
    print_every = 1
    save_every = 10

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    
    model.train()

    for epoch in range(num_epochs):
        
        running_loss = 0.0
        top_1_accuracy = 0.0
        top_5_accuracy = 0.0

        dataset = ChessDataset(file_name="../data/trainingsmall.pgn", tokenizer=tok, max_seq_len=config["seq_len"])
        dataloader = DataLoader(dataset, batch_size=batch_size)

        for i, (cur_token_ids, cur_attn_mask, cur_legal_mask, cur_labels) in enumerate(dataloader):
            optimizer.zero_grad()
            
            cur_attn_mask = cur_attn_mask.to(gpu_device)
            cur_token_ids = cur_token_ids.to(gpu_device)
            cur_legal_mask = cur_legal_mask.to(gpu_device)
            cur_labels = cur_labels.to(gpu_device)

            outputs = model(input_ids = cur_token_ids, attention_mask = cur_attn_mask).logits
            masked_logits = outputs.masked_fill(~cur_legal_mask, float('-1e10'))
            vocab_size = masked_logits.size(-1)
            
            top_1_accuracy += compute_topk_accuracy(masked_logits, cur_labels, cur_attn_mask, k=1)
            top_5_accuracy += compute_topk_accuracy(masked_logits, cur_labels, cur_attn_mask, k=5)

            masked_logits = masked_logits.view(-1, vocab_size)
            labels_flat = cur_labels.view(-1)
            loss = loss_fn(masked_logits, labels_flat)
            running_loss += loss.item()

            del outputs, cur_token_ids, cur_attn_mask, cur_legal_mask, cur_labels

            gc.collect()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # running_loss += loss.item()
            if (epoch + 1) % 1 == 0 and (i + 1) % print_every == 0:
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {loss / print_every:.4f}, Top-1 Accuracy: {top_1_accuracy / print_every:.4f}, Top-5 Accuracy: {top_5_accuracy / print_every:.4f}')
                running_loss = 0.0
                top_1_accuracy = 0.0
                top_5_accuracy = 0.0
            
            if (epoch + 1) % save_every == 0:
                torch.save(model.state_dict(), f"model_{epoch+1}.pt")
                

