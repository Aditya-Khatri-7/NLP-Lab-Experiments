# NLP Lab Experiments — Aditya Khatri (500119453)

Computational Linguistics and Natural Language Processing Lab (CSEG4040P).
Every script prints `aditya khatri 500119453` in its banner and output lines, and every figure caption carries the same line.

| Exp | Topic | Script | Key output |
|---|---|---|---|
| 1 | Employee CSV – average salary by department | `Exp1/exp1_employee_salary.py` | bar / box / scatter plots |
| 2 | Text retrieval (first 100 tweets) + word & sentence tokenization | `Exp2/exp2_text_retrieval_tokenization.py` | tweet files, tokenizer comparison |
| 3 | Pre-processing (stop-words, stemming, lemmatization) + morphology | `Exp3/exp3_processing_data.py` | cleaned data, morphology table |
| 4 | Text mining: word count, top-10, word cloud, Brown corpus, NER | `Exp4/exp4_text_mining_ner.py` | word cloud, NER highlight |
| 5 | POS tagging of tweets, chunking, HMM–Viterbi tagger | `Exp5/exp5_pos_tagging_chunking.py` | tag charts, chunk diagram (HMM ≈ 92 %) |
| 6 | N-gram LM + smoothing (Laplace, add-k, interpolation, Kneser-Ney, backoff) | `Exp6/exp6_ngram_smoothing.py` | perplexity comparison |
| 7 | Sentiment & subjectivity, IMDB classification | `Exp7/exp7_sentiment_analysis.py` | ROC, confusion matrices (LR ≈ 86 %) |
| 8 | Sentence meaning (WordNet, Lesk, similarity) + summarization | `Exp8/exp8_text_summarization.py` | TextRank / TF-IDF / frequency summaries |
| 9 & 10 | Mini-project: Movie Review Intelligence System | `Exp9_10_MiniProject/exp9_10_review_intelligence.py` | search, QA, sentiment, summary, topics |

Each experiment folder has an `outputs/` directory with its figures, CSVs and `console_output.txt`.

## Run
```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk wordcloud textblob
python Exp1/exp1_employee_salary.py        # ...and so on; shared helper is common.py
```
NLTK corpora (twitter_samples, movie_reviews, brown, treebank, …) are downloaded automatically on first run.

## Notes
* Datasets are NLTK's built-in corpora (real tweets, 2000 IMDB reviews, Brown), so no Kaggle login is needed. If `IMDB Dataset.csv` is placed in Exp3/Exp7 it is used instead.
* Exp 2 uses stored tweets because the Twitter/X API needs paid credentials; a `tweepy` function is included.
* NER/POS/chunking use NLTK (spaCy was blocked on the development laptop). NER adds a small gazetteer to fix NLTK's known mistakes.
* Exp 8: Lesk picks the wrong sense for "bass" (fish) — a known limitation, discussed in the report.
