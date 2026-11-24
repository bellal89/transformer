"""Minimal transformer: parameters + forward pass."""

from pathlib import Path

import torch
import torch.nn.functional as F

from data import BLOCK_SIZE, DATASET, get_data_batch, vocab_size, encode, decode

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

PARAM_NAMES = ["token_embedding_table", "position_embedding_table", "W_q", "W_k", "W_v", "W_out"]
# One checkpoint per dataset: vocab depends on the dataset
CHECKPOINT = Path(__file__).parent / "checkpoints" / f"{Path(DATASET).stem}.pt"


def save(path: Path = CHECKPOINT) -> None:
    """Save trained parameters."""
    path.parent.mkdir(exist_ok=True)
    torch.save({name: globals()[name] for name in PARAM_NAMES}, path)


@torch.no_grad()
def load(path: Path = CHECKPOINT) -> None:
    """Load trained parameters in place."""
    state = torch.load(path)
    for name in PARAM_NAMES:
        globals()[name].copy_(state[name])


def forward(x: torch.Tensor) -> torch.Tensor:
    """x: a batch of B sequences, T token ids each; [B, T] -> logits [B, T, vocab_size]."""
    # x: [B, T]
    B, T = x.shape

    # token_embeddings: [B, T, n_embd]
    token_embeddings = token_embedding_table[x]

    # pos: [T]
    pos = torch.arange(T)
    # pos_embeddings: [T, n_embd]
    pos_embeddings = position_embedding_table[pos]
    # embeddings: [B, T, n_embd] + [T, n_embd] = [B, T, n_embd] (broadcasting)
    embeddings = token_embeddings + pos_embeddings

    # q, k, v: [B, T, n_embd] @ [n_embd, n_embd] = [B, T, n_embd] (broadcasting)
    q = embeddings @ W_q
    k = embeddings @ W_k
    v = embeddings @ W_v

    # k_transposed: [B, n_embd, T]
    k_transposed = torch.transpose(k, 1, 2)
    # attention_scores: [B, T, T]
    attention_scores = q @ k_transposed / n_embd_sqrt

    # look_ahead_mask: [T, T]
    # prevents looking ahead, must be applied before softmax
    look_ahead_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    attention_scores = attention_scores.masked_fill(look_ahead_mask, float("-inf"))

    # subtract row max to avoid exp overflow (softmax is shift-invariant)
    attention_scores = attention_scores - attention_scores.max(dim=-1, keepdim=True).values

    # softmax over the last dim: e^x_i / sum_j e^x_j  (each row sums to 1)
    # attention_weights: [B, T, T]
    attention_weights = torch.exp(attention_scores)
    attention_weights = attention_weights / torch.sum(attention_weights, dim=-1, keepdim=True)

    # for the current token attention helps to focus the model on the most important prev tokens (incl. the current one)
    #     so the more important tokens go to the context with higher weights
    # W_q, W_k decide which tokens to look at; W_v decides what info each token passes
    #     (standard transformers have many attention heads, each with its own W_q, W_k, W_v)
    # context: [B, T, T] @ [B, T, n_embd] = [B, T, n_embd]
    context = attention_weights @ v

    # logits: [B, T, n_embd] @ [n_embd, vocab_size] = [B, T, vocab_size] (broadcasting)
    return context @ W_out


@torch.no_grad()
def generate(prefix: str, n: int, temperature: float = 0.0) -> str:
    answer = prefix
    for i in range(n):
        block = encode(answer)[-BLOCK_SIZE:]
        x = torch.tensor(block).unsqueeze(0)
        logits = forward(x)[0, -1]
        if temperature <= 0:
            y = int(logits.argmax())
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            y = int(torch.multinomial(probs, 1)[0])
        answer += decode([y])
    return answer


if __name__ == "__main__":
    x, y = get_data_batch(torch.tensor([0]))  # batch of one chunk: [1, T]
    logits = forward(x).view(-1, vocab_size)
    loss = F.cross_entropy(logits, y.view(-1))

    print("logits.shape:", logits.shape)
    print("loss:", loss.item())
    assert logits.shape == (x.numel(), vocab_size)
    assert abs(loss.item() - 2.56) < 0.01, "loss is unexpected"
    print("OK")
