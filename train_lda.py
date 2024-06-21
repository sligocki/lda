# Train LDA model on corpus

import argparse
import json
from pathlib import Path

import gensim

from utils import log


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--in-json", type=Path, default=Path("data/words_clean.json"))
  parser.add_argument("--out-model", type=Path, default=Path("data/lda_model"))

  parser.add_argument("--max-frac", type=float, default=0.5,
                      help="Maximum fraction of documents a word can appear in to be included.")
  parser.add_argument("--min-docs", type=int, default=5,
                      help="Minimum number of documents a word must appear in to be included.")
  
  parser.add_argument("--num-topics", type=int, default=5,
                      help="Number of topics for LDA algorithm to split documents into.")
  parser.add_argument("--num-passes", type=int, default=50)
  parser.add_argument("--num-interations", type=int, default=50)
  parser.add_argument("--seed", type=int, default=138,
                      help="Random seed for LDA algorithm.")
  args = parser.parse_args()

  with open(args.in_json, "r") as f:
    docs = json.load(f)
  
  id2word = gensim.corpora.Dictionary(docs.values())
  # Filter out the most common and uncommon words.
  id2word.filter_extremes(no_below=5, no_above=0.5)
  bow_corpus = {doc_id: id2word.doc2bow(doc) for doc_id, doc in docs.items()}
  corpus_values = list(bow_corpus.values())
  texts = list(docs.values())
  log(f"Loaded {len(bow_corpus):_d} documents with {len(id2word):_d} unique words")
  
  log(f"Training LDA model with {args.num_topics:_d} topics, {args.num_passes:_d} passes, and {args.num_interations:_d} iterations.")
  lda_model = gensim.models.LdaModel(corpus_values, id2word=id2word, alpha="auto",
    random_state=args.seed, num_topics=args.num_topics, passes=args.num_passes, iterations=args.num_interations)

  coherence_model = gensim.models.CoherenceModel(lda_model, texts=texts, coherence="c_v")
  log(f"C_v coherence: {coherence_model.get_coherence():.6f}")

  lda_model.save(str(args.out_model))
  log(f"Saved model to {args.out_model}")

if __name__ == "__main__":
  main()