"""Generate text with a trained model: uv run python predict.py "hel" -n 30 -t 0.5"""

import argparse

import model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("prefix", help="text to continue")
parser.add_argument("-n", type=int, default=30, help="number of characters to generate")
parser.add_argument("-t", "--temperature", type=float, default=0.0, help="0 = greedy")
args = parser.parse_args()

model.load()
print(model.generate(args.prefix, args.n, temperature=args.temperature))
