# Find what proportion of articles had each word in dataset and remove the words
# that appear too high or low proportion of the time.

import argparse
import collections
import json
from pathlib import Path

import pandas as pd


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--in-json", type=Path, default=Path("data/words_clean.json"))
  parser.add_argument("--out-json", type=Path, default=Path("data/words_filtered.json"))
  parser.add_argument("--word-freqs-csv", type=Path, default=Path("data/word_freqs.csv"))

  parser.add_argument("--max-frac", type=float, default=0.5)
  parser.add_argument("--min-frac", type=float, default=0.01)
  args = parser.parse_args()

  with open(args.in_json, "r") as f:
    docs = json.load(f)

  # Count how many documents each word appears in.
  word_counts = collections.Counter()
  for doc in docs.values():
    word_counts.update(set(doc))

  df = pd.DataFrame.from_records(word_counts.most_common(),
                                 columns = ["word", "num_docs"])
  df["frac_docs"] = df.num_docs / len(docs)
  df.to_csv(args.word_freqs_csv, index=False)

  # Filter out high and low occurence words.
  max_count = args.max_frac * len(docs)
  min_count = args.min_frac * len(docs)

  good_words = set()
  num_high_words = 0
  num_low_words = 0
  for word, count in word_counts.items():
    if count > max_count:
      num_high_words += 1
    elif count < min_count:
      num_low_words += 1
    else:
      good_words.add(word)

  print(f"Num docs: {len(docs):_d}")
  print(f"Total unique words: {len(word_counts):_d}")
  print(f"Num high frequency words removed: {num_high_words:_d}")
  print(f"Num low frequency words removed: {num_low_words:_d}")
  print(f"Remaining words: {len(good_words):_d}")

  out_docs = {}
  for id, words in docs.items():
    out_docs[id] = [word for word in words
                    if word in good_words]

main()
