import torch

def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

def top_k_accuracy(logits, labels, k=5):
    topk_values, topk_indices = torch.topk(logits, k, dim=-1)  # [batch_size, k]
    correct_in_topk = (topk_indices == labels.unsqueeze(-1)).any(dim=-1)
    topk_accuracy = correct_in_topk.float().mean().item()
    return topk_accuracy

def compute_topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor, pad_token_id: int = 0, k: int = 1) -> float:
    batch_size, seq_len, vocab_size = logits.shape

    # 1) Flatten all except vocab
    logits_flat = logits.view(-1, vocab_size)         # shape: [batch_size*seq_len, vocab_size]
    labels_flat = labels.view(-1)                     # shape: [batch_size*seq_len]
    mask_flat = attention_mask.view(-1).bool()        # shape: [batch_size*seq_len], True=real token, False=pad

    # 2) Further exclude positions where label == pad_token_id
    #    We'll combine both conditions into a single boolean mask
    valid_positions = mask_flat & (labels_flat != pad_token_id)

    if valid_positions.sum() == 0:
        # If there are no valid positions, return 0 or NaN
        return 0.0

    valid_logits = logits_flat[valid_positions]       # shape: [#valid, vocab_size]
    valid_labels = labels_flat[valid_positions]       # shape: [#valid]

    # 3) Get top-k indices along vocab dimension
    topk_values, topk_indices = torch.topk(valid_logits, k, dim=-1)  # [#valid, k]

    # 4) Check if the correct label is in top-k predictions
    correct_in_topk = (topk_indices == valid_labels.unsqueeze(-1)).any(dim=-1)  # [#valid] bool
    accuracy = correct_in_topk.float().mean().item()

    return accuracy