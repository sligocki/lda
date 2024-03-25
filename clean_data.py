# List top ngrams (combinations of adjacent words).

import argparse
import json
from pathlib import Path
import re

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

kStopWords = frozenset(stopwords.words("english"))
stemmer = SnowballStemmer("english")


def clean_word(word : str) -> str | None:
  # Remove words in all uppercase.
  if word == word.upper():
    return

  # Remove URLs
  if "://" in word or "www" in word or "http" in word:
    return

  # Convert all words to lowercase.
  word = word.lower()
  # Remove all symbols (non-letters).
  # TODO: This will also remove all accented letters, etc. Consider this.
  word = re.sub("[^a-z]", "", word)
  # Lemmatize (remove inflection endings. Ex: testing -> test)
  word = stemmer.stem(word)

  # Remove stop words (a, the, ...)
  if word in kStopWords:
    return

  return word

def clean_words(in_words : list[str]) -> list[str]:
  # Clean individual words
  out_words = []
  for word in in_words:
    word = clean_word(word)
    if word:
      out_words.append(word)

  # TODO: Remove words from denylist

  # TODO: Combine ngrams

  return out_words


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--in-json", type=Path, default=Path("data/words_raw.json"))
  parser.add_argument("--out-json", type=Path, default=Path("data/words_clean.json"))
  parser.add_argument("--denylist", type=Path, default=Path("data/input/denylist.txt"),
                      help="List of words/phrases to remove from corpus (Journal titles, etc.). One per line.")
  args = parser.parse_args()

  with open(args.in_json, "r") as f:
    in_docs = json.load(f)

  out_docs = {id: clean_words(words)
              for id, words in in_docs.items()}

  with open(args.out_json, "w") as f:
    json.dump(out_docs, f)

main()
