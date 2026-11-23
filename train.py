"""Training loop: forward -> loss -> gradients -> update weights, repeated."""

import torch
import torch.nn.functional as functions

import model
from data import BLOCK_SIZE, data, get_data_batch

LEARNING_RATE = 0.1
STEPS = 10000
BATCH_SIZE = 16

params = [
    model.token_embedding_table,
    model.position_embedding_table,
    model.W_q,
    model.W_k,
    model.W_v,
    model.W_out,
]

# make PyTorch track gradients for each parameter
for p in params:
    p.requires_grad_(True)


# flattens batch into shape ([B x T, vocab_size], [B x T]) then averages B x T losses
def get_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return functions.cross_entropy(
            logits.view(-1, model.vocab_size),
            target.view(-1),
            reduction='mean'
        )


@torch.no_grad()
def estimate_loss() -> torch.Tensor:
    x, y = get_data_batch(torch.arange(len(data) - BLOCK_SIZE))
    logits = model.forward(x)
    return get_loss(logits, y)


for step in range(STEPS + 1):
    # batch of random text chunks to smooth the train loss curve
    random_ids = torch.randint(0, len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    batch_x, batch_y = get_data_batch(random_ids)

    # forward pass and loss
    batch_logits = model.forward(batch_x)
    train_loss = get_loss(batch_logits, batch_y)

    # reset old gradients, or otherwise backward adds new gradients to the old ones
    for p in params:
        p.grad = None

    # compute gradients
    train_loss.backward()

    # Adjust weight matrices: move each parameter against its gradient
    # Do it in place (p -= ...) inside `with torch.no_grad()`, so the update itself is not recorded in the autograd graph
    with torch.no_grad():
        for p in params:
            p -= LEARNING_RATE * p.grad

    if step % 250 == 0:
        est_loss = estimate_loss()
        print(f"step {step:5d}   loss {est_loss:.3f}")
