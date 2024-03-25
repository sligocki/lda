# Find what proportion of articles had each word in dataset.

import argparse
import collections
import json
from pathlib import Path

import pandas as pd


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--in-json", type=Path, default=Path("data/words_clean.json"))
  parser.add_argument("--word-freqs-csv", type=Path, default=Path("data/word_freqs.csv"))
  parser.add_argument("--max-ngram-size", type=int, default=10)
  parser.add_argument("--num-top", type=int, default=1000,
                      help="How many top ngrams to list for each n.")
  args = parser.parse_args()

  with open(args.in_json, "r") as f:
    docs = json.load(f)

  word_counts = collections.Counter()
  for doc in docs.values():
    word_counts.update(set(doc))

  df = pd.DataFrame.from_records(word_counts.most_common(),
                                 columns = ["word", "num_docs"])
  df["frac_docs"] = df.num_docs / len(docs)
  df.to_csv(args.word_freqs_csv, index=False)

main()
