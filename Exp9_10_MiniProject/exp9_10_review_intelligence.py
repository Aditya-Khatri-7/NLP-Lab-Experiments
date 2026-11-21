"""EXPERIMENT 9 & 10: MINI-PROJECT - Movie Review Intelligence System
Author: aditya khatri 500119453

One pipeline combining several NLP applications from the syllabus:
 1. Semantic search / information retrieval (TF-IDF + cosine)
 2. Sentiment classification (TF-IDF + Logistic Regression)
 3. Extractive summarization (TextRank)
 4. Keyword extraction (top TF-IDF terms)
 5. Topic discovery (K-Means clustering + LSA/PCA map)
 6. Simple question-answering over the reviews (retrieve best sentence)
Data: NLTK movie_reviews (2000 IMDB reviews). Run:  python exp9_10_review_intelligence.py ["your query"]
"""
import os, sys, re, textwrap
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # avoids K-Means hang on Windows
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, nltk
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
from matplotlib.patches import FancyBboxPatch
from common import *

banner("EXPERIMENT 9 & 10 - MINI PROJECT: Movie Review Intelligence System")
OUT = out_dir(__file__)
for r in ["movie_reviews", "stopwords", "punkt", "punkt_tab", "averaged_perceptron_tagger_eng", "maxent_ne_chunker_tab", "words"]: nltk.download(r, quiet=True)
stop = set(stopwords.words("english"))
fids = movie_reviews.fileids()
df = pd.DataFrame({"id": fids, "text": [re.sub(r"\s+", " ", movie_reviews.raw(f)) for f in fids], "label": [int(f.startswith("pos")) for f in fids]})
say("corpus:", len(df), "reviews")

# ---- 1. retrieval ----
vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_df=.8, sublinear_tf=True, max_features=30000); X = vec.fit_transform(df.text)
def search(q, k=5):
    s = cosine_similarity(vec.transform([q]), X).ravel(); i = np.argsort(s)[::-1][:k]; return [(df.id[j], s[j], j) for j in i]

# ---- 2. sentiment ----
Xtr, Xte, ytr, yte = train_test_split(np.arange(len(df)), df.label, test_size=.2, stratify=df.label, random_state=500119453)
clf = LogisticRegression(max_iter=2000, C=5).fit(X[Xtr], ytr); acc = accuracy_score(yte, clf.predict(X[Xte]))
say(f"sentiment classifier test accuracy: {acc*100:.2f}%")
def sentiment(j): p = clf.predict_proba(X[j])[0, 1]; return ("POSITIVE" if p > .5 else "NEGATIVE", p)

# ---- 3. summarization (TextRank) ----
def summarize(text, k=2):
    S = [s for s in sent_tokenize(text) if len(s.split()) >= 5]
    if len(S) <= k: return S
    M = TfidfVectorizer(stop_words="english").fit_transform(S); A = cosine_similarity(M); np.fill_diagonal(A, 0)
    rs = A.sum(1, keepdims=True); P = np.divide(A, rs, out=np.zeros_like(A), where=rs > 0); r = np.ones(len(S)) / len(S)
    for _ in range(40): r = .15 / len(S) + .85 * P.T @ r
    return [S[i] for i in sorted(np.argsort(r)[::-1][:k])]

# ---- 4. keywords + entities ----
def keywords(j, k=6):
    row = X[j].toarray().ravel(); names = vec.get_feature_names_out(); return [names[i] for i in np.argsort(row)[::-1][:k]]

# ---- 6. QA ----
def answer(question, pool=30):
    hits = search(question, pool); sents = [(s, h[0]) for h in hits for s in sent_tokenize(df.text[h[2]]) if len(s.split()) > 5]
    v = TfidfVectorizer(stop_words="english").fit([question] + [s for s, _ in sents]); sc = cosine_similarity(v.transform([question]), v.transform([s for s, _ in sents])).ravel()
    b = int(np.argmax(sc)); return sents[b][0], sents[b][1], sc[b]

# ================= demo =================
queries = sys.argv[1:] or ["a thrilling space adventure with aliens", "romantic comedy with great chemistry", "boring plot and terrible acting"]
section("Search + Sentiment + Summary + Keywords")
for q in queries:
    say(f"QUERY: {q}")
    for rank, (fid, sc, j) in enumerate(search(q, 3), 1):
        lab, p = sentiment(j)
        say(f" #{rank} {fid} (similarity {sc:.3f}) | sentiment {lab} ({p:.2f}) | true: {'POSITIVE' if df.label[j] else 'NEGATIVE'}")
        say("    keywords:", keywords(j)); say("    summary :", " ".join(summarize(df.text[j])))
section("Question answering")
for qn in ["Who directed the film?", "What was wrong with the special effects?", "Why is the acting praised?"]:
    a, f, s = answer(qn); say(f"Q: {qn}"); say(f"A: {a[:230]}   [from {f}, score {s:.2f}]")

# ---- 5. topic discovery ----
section("Topic discovery with K-Means")
GENERIC = {"movie", "film", "like", "just", "really", "good", "bad", "time", "make", "get", "one", "even", "story", "movies", "films"}
K = 6; svd = TruncatedSVD(60, random_state=1); L = svd.fit_transform(X); L /= np.linalg.norm(L, axis=1, keepdims=True) + 1e-9
km = KMeans(K, n_init=5, random_state=1).fit(L); names = vec.get_feature_names_out(); topics = {}
for c in range(K):
    cen = X[km.labels_ == c].mean(axis=0).A1; kw = [names[i] for i in np.argsort(cen)[::-1] if names[i] not in GENERIC][:6]; topics[c] = kw
    say(f"topic {c} ({(km.labels_==c).sum():4d} reviews, {df.label[km.labels_==c].mean()*100:3.0f}% positive):", kw)
Z = L[:, :2]

# ================= figures =================
fig, ax = plt.subplots(figsize=(9, 6.5))
for c in range(K): m = km.labels_ == c; ax.scatter(Z[m, 0], Z[m, 1], s=10, alpha=.6, color=PAL[c], label=f"T{c}: " + ", ".join(topics[c][:3]))
ax.set(title="Topic Map of 2000 Reviews (LSA projection + K-Means)", xlabel="component 1", ylabel="component 2"); ax.legend(fontsize=8, markerscale=2)
save(fig, OUT, "topic_map.png", "Fig 1: Review clusters")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
share = [df.label[km.labels_ == c].mean() * 100 for c in range(K)]; axs[0].bar([f"T{c}" for c in range(K)], share, color=PAL[:K]); axs[0].axhline(50, ls="--", c="k", lw=1)
axs[0].set(title="% Positive Reviews per Topic", ylabel="% positive", ylim=(0, 100))
axs[1].bar([f"T{c}" for c in range(K)], [(km.labels_ == c).sum() for c in range(K)], color=PAL[:K]); axs[1].set(title="Reviews per Topic", ylabel="count")
save(fig, OUT, "topic_sentiment.png", "Fig 2: Sentiment and size of each topic")

q = queries[0]; hits = search(q, 8)
fig, ax = plt.subplots(figsize=(9, 5)); cols = [PAL[2] if sentiment(h[2])[0] == "POSITIVE" else PAL[3] for h in hits]
ax.barh([h[0].split("/")[1][:14] for h in hits][::-1], [h[1] for h in hits][::-1], color=cols[::-1]); ax.set(title=f'Search results for: "{q}"', xlabel="cosine similarity")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PAL[2]), plt.Rectangle((0, 0), 1, 1, color=PAL[3])], labels=["positive", "negative"], loc="lower right")
save(fig, OUT, "search_results.png", "Fig 3: Retrieval ranking with predicted sentiment")

names_arr = np.array(vec.get_feature_names_out()); co = np.argsort(clf.coef_[0])
fig, ax = plt.subplots(figsize=(9, 6)); idx = np.r_[co[:10], co[-10:]]
ax.barh(names_arr[idx], clf.coef_[0][idx], color=[PAL[3] if v < 0 else PAL[2] for v in clf.coef_[0][idx]]); ax.set(title=f"Sentiment Model Evidence (test accuracy {acc*100:.1f}%)", xlabel="weight")
save(fig, OUT, "sentiment_evidence.png", "Fig 4: Words the classifier trusts")

fig, ax = plt.subplots(figsize=(11, 6.2)); ax.axis("off"); ax.grid(False)
ax.set_title("System Architecture - Movie Review Intelligence", fontsize=14, fontweight="bold")
boxes = [("Raw reviews\n(IMDB corpus)", .02, .45), ("TF-IDF\nvectoriser", .22, .45), ("Search /\nQA", .45, .75), ("Sentiment\nclassifier", .45, .5), ("TextRank\nsummary", .45, .25), ("K-Means\ntopics", .45, .0), ("Report for\nuser", .75, .4)]
for t, x, y in boxes:
    ax.add_patch(FancyBboxPatch((x, y), .16, .16, boxstyle="round,pad=.01", fc="#dfe9f7", ec=PAL[0], transform=ax.transAxes, clip_on=False)); ax.text(x + .08, y + .08, t, ha="center", va="center", transform=ax.transAxes, fontsize=10)
for (x1, y1, x2, y2) in [(.18, .53, .22, .53), (.38, .53, .45, .83), (.38, .53, .45, .58), (.38, .53, .45, .33), (.38, .53, .45, .08), (.61, .83, .75, .5), (.61, .58, .75, .5), (.61, .33, .75, .46), (.61, .08, .75, .43)]:
    ax.annotate("", (x2, y2), (x1, y1), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", color="#555"))
save(fig, OUT, "architecture.png", "Fig 5: Pipeline of the mini-project")
say("EXPERIMENT 9 & 10 COMPLETED")
