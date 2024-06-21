# Describe details about this LDA model


import argparse
import json
from pathlib import Path

import gensim
import pandas as pd

from utils import log


# gensim "bag of words" type.
BOW = list[tuple[int, float]]


def doc_topics(lda_model : gensim.models.LdaModel, bow_corpus : dict[str, BOW]) -> pd.DataFrame:
  """Load data frame with topic weights for each document."""
  rows = []
  for doc_id, bow in bow_corpus.items():
    weights = lda_model.get_document_topics(bow)
    row = dict(weights)
    row["cov_num"] = doc_id
    rows.append(row)
  df = pd.DataFrame(rows, columns=["cov_num"] + list(range(lda_model.num_topics)))
  return df.set_index("cov_num")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model", type=Path)
  parser.add_argument("--words-json", type=Path, default=Path("data/words_clean.json"))
  args = parser.parse_args()

  lda_model = gensim.models.LdaModel.load(str(args.model))
  log(f"Loaded LDA model from {args.model}")

  with open(args.words_json, "r") as f:
    docs = json.load(f)
  bow_corpus_dict = {doc_id: lda_model.id2word.doc2bow(doc) for doc_id, doc in docs.items()}
  bow_corpus = list(bow_corpus_dict.values())
  texts = list(docs.values())

  log("Top words per topic:")
  for topic_id in range(lda_model.num_topics):
    print(f"  Topic {topic_id}")
    for word, weight in lda_model.show_topic(topic_id):
      print(f"    {word:<15} {weight:.6f}")

  log("Top documents per topic:")
  doc_topics_df = doc_topics(lda_model, bow_corpus_dict)
  for topic_id in range(lda_model.num_topics):
    print(f"  Topic {topic_id}")
    print(doc_topics_df.sort_values(topic_id, ascending=False)[topic_id].head(10))

  log(f"Perplexity: {lda_model.log_perplexity(bow_corpus):.6f}")
  
  coherence_model = gensim.models.CoherenceModel(lda_model, texts=texts, coherence="c_v")
  log(f"C_v coherence: {coherence_model.get_coherence():.6f}")

if __name__ == "__main__":
  main()