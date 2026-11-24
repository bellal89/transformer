# transformer

A minimal character-level transformer written from scratch in PyTorch
(tensors and autograd only, no `nn` modules): one self-attention head,
learned token and position embeddings, trained on next-character prediction.

## Usage

```sh
uv sync
make train                                  # train on datasets/hello.txt
make predict                                # continue "h" with 30 chars
uv run python predict.py "wor" -n 50 -t 0.8 # custom prefix, length, temperature
```

Another dataset: put a text file into `datasets/` and pass its name in `DATASET`,
both for training and prediction:

```sh
DATASET=my.txt make train
DATASET=my.txt make predict ARGS='"a" -n 100'
```
