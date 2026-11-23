"""Minimal transformer: parameters + forward pass."""

import torch
import torch.nn.functional as F

from data import BLOCK_SIZE, get_data_batch, vocab_size

torch.manual_seed(42)

n_embd = 16
n_embd_sqrt = n_embd ** 0.5

# --- Parameters: everything the model learns. Created once. ---

token_embedding_table = torch.randn(vocab_size, n_embd)
position_embedding_table = torch.randn(BLOCK_SIZE, n_embd)
W_q = torch.randn(n_embd, n_embd) / n_embd_sqrt
W_k = torch.randn(n_embd, n_embd) / n_embd_sqrt
W_v = torch.randn(n_embd, n_embd) / n_embd_sqrt
W_out = torch.randn(n_embd, vocab_size) / n_embd_sqrt


def forward(x: torch.Tensor) -> torch.Tensor:
    """x: B BATCH_SIZE of T token ids  [B, T] -> logits [B, T, vocab_size]."""
    # x: [B, T]
    B, T = x.shape

    # token_embedding: [B, T, n_embd]
    token_embedding = token_embedding_table[x]

    # pos: [T]
    pos = torch.arange(T)
    # pos_embedding: [T, n_embd]
    pos_embedding = position_embedding_table[pos]
    # h: [B, T, n_embd] + [T, n_embd] = [B, T, n_embd] (broadcasting)
    h = token_embedding + pos_embedding

    # q, k, v: [B, T, n_embd] @ [n_embd, n_embd] = [B, T, n_embd] (broadcasting)
    q = h @ W_q
    k = h @ W_k
    v = h @ W_v

    # k_transposed: [B, n_embd, T]
    k_transposed = torch.transpose(k, 1, 2)
    # scores: [B, T, T]
    scores = q @ k_transposed / n_embd_sqrt
    # mask: [T, T]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    scores = scores - scores.max(dim=-1, keepdim=True).values

    # softmax weights: [B, T, T]
    weights = torch.exp(scores)
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)

    # out: [B, T, T] @ [B, T, n_embd] = [B, T, n_embd]
    out = weights @ v

    # logits: [B, T, n_embd] @ [n_embd, vocab_size] = [B, T, vocab_size] (broadcasting)
    return out @ W_out


if __name__ == "__main__":
    x, y = get_data_batch(torch.tensor([0]))  # batch of one chunk: [1, T]
    logits = forward(x).view(-1, vocab_size)
    loss = F.cross_entropy(logits, y.view(-1))

    print("logits.shape:", logits.shape)
    print("loss:", loss.item())
    assert logits.shape == (x.numel(), vocab_size)
    assert abs(loss.item() - 2.56) < 0.01, "loss is unexpected"
    print("OK")
