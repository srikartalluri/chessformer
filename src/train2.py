import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import torch.optim as optim

import gc

from metrics import *
from utils import setup
from transformers import get_linear_schedule_with_warmup
import argparse

# from eval_play import play_n
# from engine import Engine




def main():
    

    model, tok, config, gpu_device, chess_dataset = setup()

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for DataLoader')
    args = parser.parse_args()
    batch_size = args.batch_size if args.batch_size else config["batch_size"]


    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    num_epochs = config["num_epochs"]
    print_every = config["print_every"]
    save_every = config["save_every"]
    eval_every = config["eval_every"]

    dataloader = DataLoader(chess_dataset, batch_size=batch_size)
    num_training_steps = 3348302 // batch_size

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps
    )

    model.train()

    for epoch in range(num_epochs):
        
        running_loss = 0.0
        top_1_accuracy = 0.0
        top_5_accuracy = 0.0

        # dataset = ChessDataset(file_name=config["training_file"], tokenizer=tok, max_seq_len=config["seq_len"])
        dataloader = DataLoader(chess_dataset, batch_size=batch_size)

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
            scheduler.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]

            # running_loss += loss.item()
            if (epoch + 1) % 1 == 0 and (i + 1) % print_every == 0:
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {loss / print_every:.4f}, Top-1 Accuracy: {top_1_accuracy / print_every:.4f}, Top-5 Accuracy: {top_5_accuracy / print_every:.4f}')
                running_loss = 0.0
                top_1_accuracy = 0.0
                top_5_accuracy = 0.0
            
        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), f"./models/model_{epoch+1}.pt")
            print("model saved")



if __name__ == "__main__":
    main()