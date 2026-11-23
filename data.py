"""Character-level data for a minimal transformer."""

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
block_size = 8  # how many tokens the model sees in one pass


def get_example_train_set(i: int) -> tuple[list[int], list[int]]:
    """Return (x, y) starting at position i in data.

    x: block_size tokens the model sees;
    y: the same tokens shifted by 1 (targets to predict).
    """
    assert len(data) > i + block_size
    x = data[i: i+block_size]
    y = data[i+1: i+block_size+1]
    return x, y


if __name__ == "__main__":
    print(chars, vocab_size)
    print(stoi, itos)
    assert decode(encode("hello")) == "hello"
    x, y = get_example_train_set(0)
    print("x:", x, repr(decode(x)))
    print("y:", y, repr(decode(y)))
