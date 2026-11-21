"""EXPERIMENT 8: Meaning of sentences (WordNet, WSD, similarity) + extractive text summarization of IMDB reviews.
Author: aditya khatri 500119453
Summarizers (from scratch): word-frequency, TF-IDF sentence scoring, TextRank (graph PageRank), Lead-N baseline.
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, nltk
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import movie_reviews, stopwords, wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.wsd import lesk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from common import *

banner("EXPERIMENT 8 - Text Summarization and Meaning of Sentences")
OUT = out_dir(__file__)
for r in ["movie_reviews", "stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]: nltk.download(r, quiet=True)
stop = set(stopwords.words("english"))

# ================= Part 1: analysing meaning of sentences =================
section("Part 1a: word senses (WordNet)")
for w in ["bank", "light"]:
    ss = wn.synsets(w); say(f"'{w}' has {len(ss)} senses; first 4:")
    for s in ss[:4]: say(f"   {s.name():16s} {s.definition()}")
section("Part 1b: word sense disambiguation (Lesk)")
wsd = [("I went to the bank to deposit my money", "bank"), ("The fisherman sat on the bank of the river", "bank"),
       ("He plays the bass guitar in a band", "bass"), ("We caught a large bass in the lake", "bass")]
for sent, w in wsd:
    s = lesk(word_tokenize(sent), w); say(f"'{w}' in '{sent}' ->", s.name() if s else None, "|", s.definition() if s else "")
section("Part 1c: sentence similarity")
pairs = [("The movie was great and entertaining", "The film was wonderful and fun"),
         ("The movie was great and entertaining", "The weather is cold in winter"),
         ("I loved the acting of the lead actor", "The lead actor's performance was loved")]
def toks(s): return {w.lower() for w in word_tokenize(s) if w.isalpha() and w.lower() not in stop}
def jacc(a, b): a, b = toks(a), toks(b); return len(a & b) / len(a | b) if a | b else 0
def wn_sim(a, b):
    sc = []
    for x in toks(a):
        best = max([(x_s.path_similarity(y_s) or 0) for x_s in wn.synsets(x)[:3] for y in toks(b) for y_s in wn.synsets(y)[:3]] or [0]); sc.append(best)
    return float(np.mean(sc)) if sc else 0
psim = []
for a, b in pairs:
    v = TfidfVectorizer().fit([a, b]); c = cosine_similarity(v.transform([a]), v.transform([b]))[0, 0]
    psim.append((a[:30], b[:30], jacc(a, b), c, wn_sim(a, b))); say(f"Jaccard={psim[-1][2]:.2f} Cosine={c:.2f} WordNet={psim[-1][4]:.2f} | '{a}' ~ '{b}'")
fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(pairs)); w = .26
for i, (n, k) in enumerate([("Jaccard", 2), ("TF-IDF cosine", 3), ("WordNet path", 4)]): ax.bar(x + i * w, [p[k] for p in psim], w, label=n, color=PAL[i])
ax.set_xticks(x + w, [f"pair {i+1}" for i in x]); ax.set(title="Sentence Similarity Measures", ylabel="similarity", ylim=(0, 1)); ax.legend()
save(fig, OUT, "sentence_similarity.png", "Fig 1: Lexical vs semantic similarity of sentence pairs")

# ================= Part 2: extractive summarization =================
def split(text): return [s.strip() for s in sent_tokenize(re.sub(r"\s+", " ", text)) if len(s.split()) >= 5]
def norm(s): return [w for w in re.findall(r"[a-z']+", s.lower()) if w not in stop and len(w) > 2]

def sum_freq(sents, k):
    f = Counter(w for s in sents for w in norm(s)); m = max(f.values())
    sc = [sum(f[w] / m for w in norm(s)) / (len(norm(s)) + 1) for s in sents]
    return sorted(np.argsort(sc)[::-1][:k]), sc
def sum_tfidf(sents, k):
    M = TfidfVectorizer(stop_words="english").fit_transform(sents); sc = np.asarray(M.sum(1)).ravel() / (np.asarray((M > 0).sum(1)).ravel() + 1)
    return sorted(np.argsort(sc)[::-1][:k]), sc
def sum_textrank(sents, k, d=.85, it=50):
    M = TfidfVectorizer(stop_words="english").fit_transform(sents); S = cosine_similarity(M); np.fill_diagonal(S, 0)
    rs = S.sum(1, keepdims=True); P = np.divide(S, rs, out=np.zeros_like(S), where=rs > 0); r = np.ones(len(sents)) / len(sents)
    for _ in range(it): r = (1 - d) / len(sents) + d * P.T @ r
    return sorted(np.argsort(r)[::-1][:k]), r
def sum_lead(sents, k): return list(range(min(k, len(sents)))), None
METHODS = {"Frequency": sum_freq, "TF-IDF": sum_tfidf, "TextRank": sum_textrank, "Lead-3": sum_lead}

section("Part 2a: summarising one review with each method")
fids = [f for f in movie_reviews.fileids() if 280 < len(movie_reviews.words(f)) < 700][:60]
doc = movie_reviews.raw(fids[0]); S = split(doc); K = 3
say("original review:", len(S), "sentences,", len(doc.split()), "words"); say("TEXT:", " ".join(S)[:600], "...")
for n, f in METHODS.items():
    idx, _ = f(S, K); say(f"[{n}]", " ".join(S[i] for i in idx))

section("Part 2b: evaluation on 60 IMDB reviews")
def rouge1(ref, hyp):
    a, b = Counter(norm(ref)), Counter(norm(hyp)); ov = sum((a & b).values())
    p = ov / max(sum(b.values()), 1); r = ov / max(sum(a.values()), 1); return 2 * p * r / (p + r) if p + r else 0
rows = []
for fid in fids:
    raw = movie_reviews.raw(fid); S = split(raw)
    if len(S) < 6: continue
    lab = fid.split("/")[0]; k = max(2, round(len(S) * .25)); full_pol = TextBlob(" ".join(S)).sentiment.polarity
    key = " ".join(w for w, _ in Counter(w for s in S for w in norm(s)).most_common(25))  # keyword "reference"
    for n, f in METHODS.items():
        idx, _ = f(S, k); summ = " ".join(S[i] for i in idx); sp = TextBlob(summ).sentiment.polarity
        rows.append((fid, n, len(summ.split()) / len(" ".join(S).split()), rouge1(" ".join(S), summ), rouge1(key, summ), np.sign(sp) == np.sign(full_pol)))
ev = pd.DataFrame(rows, columns=["doc", "method", "compression", "ROUGE1_vs_full", "keyword_overlap", "sentiment_kept"])
agg = ev.groupby("method").agg(compression=("compression", "mean"), rouge1=("ROUGE1_vs_full", "mean"), keyword=("keyword_overlap", "mean"), sentiment_kept=("sentiment_kept", "mean"))
print(agg.round(3).to_string()); agg.to_csv(os.path.join(OUT, "summarization_eval.csv"))
say("best ROUGE-1:", agg.rouge1.idxmax(), "| best keyword coverage:", agg.keyword.idxmax(), "| best sentiment preservation:", agg.sentiment_kept.idxmax())

# ---- figures ----
S = split(movie_reviews.raw(fids[0])); idx, sc = sum_textrank(S, 3)  # same sentences as Part 2a
fig, ax = plt.subplots(figsize=(11, 4.8)); ax.bar(range(1, len(S) + 1), sc, color=[PAL[2] if i in idx else "#c6cfdb" for i in range(len(S))])
ax.set(title="TextRank Sentence Importance (green = selected for summary)", xlabel="sentence number", ylabel="TextRank score")
save(fig, OUT, "textrank_scores.png", "Fig 2: Which sentences TextRank picks")

fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
for ax, c, t in zip(axs, ["rouge1", "keyword", "sentiment_kept"], ["ROUGE-1 F1 vs full review", "Keyword coverage", "Sentiment preserved"]):
    ax.bar(agg.index, agg[c], color=PAL[:4]); ax.set_title(t); plt.setp(ax.get_xticklabels(), rotation=25)
    for i, v in enumerate(agg[c]): ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
save(fig, OUT, "summarizer_comparison.png", "Fig 3: Summarizers compared on 60 IMDB reviews")

fig, ax = plt.subplots(figsize=(11, 4.2)); ax.axis("off"); ax.grid(False); y = .93
ax.text(0, y, "Review sentences (green = chosen by TextRank for the summary):", fontweight="bold"); y -= .11
for i, s_ in enumerate(S[:8]):
    t_ = s_ if len(s_) <= 130 else s_[:127] + "..."
    ax.text(0, y, f"{i+1}. {t_}", fontsize=8.5, bbox=dict(fc="#bfe3cf" if i in idx else "none", ec="none")); y -= .105
save(fig, OUT, "summary_highlight.png", "Fig 4: Extractive summary highlighted in the review")

wc_fig, ax = plt.subplots(figsize=(8, 4.8)); lens = [len(split(movie_reviews.raw(f))) for f in fids]
ax.hist(lens, bins=15, color=PAL[4], edgecolor="white"); ax.set(title="Review Length (sentences)", xlabel="sentences", ylabel="reviews")
save(wc_fig, OUT, "review_lengths.png", "Fig 5: Length of evaluated reviews")
say("EXPERIMENT 8 COMPLETED")
