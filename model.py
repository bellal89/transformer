"""Minimal transformer: parameters + forward pass."""

import torch
import torch.nn.functional as F

from data import block_size, get_example_train_set, vocab_size

torch.manual_seed(42)

n_embd = 16
n_embd_sqrt = n_embd ** 0.5

# --- Parameters: everything the model learns. Created once. ---

# [vocab_size, n_embd]
token_embedding_table = torch.randn(vocab_size, n_embd)
# [block_size, n_embd]
position_embedding_table = torch.randn(block_size, n_embd)
# [n_embd, n_embd]
W_q = torch.randn(n_embd, n_embd) / n_embd_sqrt
W_k = torch.randn(n_embd, n_embd) / n_embd_sqrt
W_v = torch.randn(n_embd, n_embd) / n_embd_sqrt
# [n_embd, vocab_size]
W_out = torch.randn(n_embd, vocab_size) / n_embd_sqrt


def forward(x: torch.Tensor) -> torch.Tensor:
    """x: token ids [T] -> logits [T, vocab_size]."""
    T = len(x)
    token_embedding = token_embedding_table[x]
    pos = torch.arange(T)
    pos_embedding = position_embedding_table[pos]
    # [T, n_embd]
    h = token_embedding + pos_embedding

    # [T, n_embd]
    q = h @ W_q
    k = h @ W_k
    v = h @ W_v

    scores = q @ k.T / n_embd_sqrt
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    scores = scores - scores.max(dim=1, keepdim=True).values

    # softmax  [T, T]
    weights = torch.exp(scores)
    weights = weights / torch.sum(weights, dim=1, keepdim=True)

    # [T, n_embd]
    out = weights @ v

    # logits: [T, vocab_size]
    return out @ W_out


if __name__ == "__main__":
    x, y = get_example_train_set(0)
    x, y = torch.tensor(x), torch.tensor(y)

    logits = forward(x)
    loss = F.cross_entropy(logits, y)

    print("logits.shape:", logits.shape)
    print("loss:", loss.item())
    assert logits.shape == (len(x), vocab_size)
    assert abs(loss.item() - 2.56) < 0.01, "loss differs from explore.py: check parameter creation order"
    print("OK")
