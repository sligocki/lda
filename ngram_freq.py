# List top ngrams (combinations of adjacent words).

import argparse
import collections
import json
from pathlib import Path

import pandas as pd


def list_top_ngrams(docs : dict[str, list[str]], n : int) -> None:
  ngrams = collections.Counter()
  for doc in docs.values():
    for i in range(len(doc) - n + 1):
      ngram = "_".join(doc[i : i+n])
      ngrams[ngram] += 1
  return ngrams

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("in_json", type=Path)
  parser.add_argument("out_dir", type=Path)
  parser.add_argument("--max-ngram-size", type=int, default=10)
  parser.add_argument("--num-top", type=int, default=1000,
                      help="How many top ngrams to list for each n.")
  args = parser.parse_args()

  args.out_dir.mkdir(exist_ok=True)

  with open(args.in_json, "r") as f:
    docs = json.load(f)

  for n in range(1, args.max_ngram_size + 1):
    ngrams = list_top_ngrams(docs, n)

    df = pd.DataFrame.from_records(ngrams.most_common(args.num_top),
                                   columns = ["ngram", "count"])
    df.to_csv(args.out_dir / f"top_ngrams_{n}.txt", index=False)

main()
