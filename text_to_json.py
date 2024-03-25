# Load all text files and convert to dict mapping id to list of all words.

import argparse
from pathlib import Path
import json


def remove_denylist(text : str, denylist : list[str]) -> str:
  """Remove all instances of any string in `denylist` from `text`."""
  for deny in denylist:
    text = text.replace(deny, "")
  return text

def load_words(dir : Path, denylist : list[str]) -> dict[str, list[str]]:
  """Load all text files in `dir` and split into list of words for each."""
  docs : dict[str, list[str]] = {}
  for path in dir.iterdir():
    id = path.stem
    text = path.read_text()
    text = remove_denylist(text, denylist)
    # Split into list of words.
    docs[id] = text.split()
  return docs

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--in-dir", type=Path, default=Path("data/input/PDFtoText/"))
  parser.add_argument("--out-json", type=Path, default=Path("data/words_raw.json"))
  parser.add_argument("--denylist", type=Path, default=Path("data/input/denylist.txt"),
                      help="List of strings to remove from corpus (Journal titles, etc.). One per line.")
  args = parser.parse_args()

  with open(args.denylist, "r") as f:
    denylist = [line.strip() for line in f if line.strip()]

  docs = load_words(args.in_dir, denylist)

  with open(args.out_json, "w") as f:
    json.dump(docs, f)

main()
