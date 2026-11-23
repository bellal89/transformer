"""Character-level data for a minimal transformer."""
import torch

text = "hello world! " * 20  # toy text for now

# Sorted list of unique tokens in text (here just symbols)
chars = sorted(set(text))
vocab_size = len(chars)

# stoi (string→int) and itos (int→string)
stoi = {char: i for i, char in enumerate(chars)}
itos = chars


def encode(s: str) -> list[int]:
    """String -> list of token ids."""
    return [stoi[char] for char in s]


def decode(ids: list[int]) -> str:
    """List of token ids -> string."""
    return "".join(itos[i] for i in ids)


# Train set
data = encode(text)
BLOCK_SIZE = 8  # how many tokens the model sees in one pass


def get_data_batch(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ids: start positions [B] -> (x, y), each [B, BLOCK_SIZE].

    x: tokens the model sees; y: the same tokens shifted by 1 (targets).
    """
    assert ids.min() >= 0 and len(data) > ids.max() + BLOCK_SIZE
    x_batch = torch.stack([torch.tensor(data[i:i + BLOCK_SIZE]) for i in ids])
    y_batch = torch.stack([torch.tensor(data[i+1: i + BLOCK_SIZE + 1]) for i in ids])
    return x_batch, y_batch


def get_data_example(i: int) -> tuple[torch.Tensor, torch.Tensor]:
    """i: start position -> (x, y), each [BLOCK_SIZE]. See get_train_batch."""
    assert 0 <= i and len(data) > i + BLOCK_SIZE
    x = torch.tensor(data[i:i + BLOCK_SIZE])
    y = torch.tensor(data[i+1: i + BLOCK_SIZE + 1])
    return x, y


if __name__ == "__main__":
    print(chars, vocab_size)
    print(stoi, itos)
    assert decode(encode("hello")) == "hello"
    x, y = get_data_example(0)
    print("x:", x, repr(decode(x.tolist())))
    print("y:", y, repr(decode(y.tolist())))
    xb, yb = get_data_batch(torch.tensor([0, 1]))
    print("batch x:", xb.shape, [decode(r.tolist()) for r in xb])
