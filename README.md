Depends upon having the following inputs:

* `data/input/PDFtoText/`
  - One text file per article with name `{id}.txt` and contents the entire text of the article.
* `data/input/denylist.txt`
  - List of strings to remove from corpus (Journal titles, etc.). One per line.
* `data/input/ngrams.txt`
  - List of ngrams to keep connected into a single token during processing (ex: high_school, activ_learn). These should be in "lemmatized" form and with _ connecting the seperate words. One ngram per line.


Example usage:

```bash
python -m pip install -r requirements.txt
python setup.py

python text_to_json.py
python clean_data.py
python train_lda.py
python describe_lda.py data/lda_model

python remove_high_low_freq.py
python ngram_freq.py data/words_clean.json data/ngrams/
```