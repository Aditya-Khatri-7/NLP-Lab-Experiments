"""EXPERIMENT 7: Sentiment analysis - polarity & subjectivity, and IMDB movie review classification.
Author: aditya khatri 500119453
Data: NLTK movie_reviews (2000 IMDB reviews, balanced). Uses IMDB Dataset.csv if placed in this folder.
Methods: TextBlob (polarity, subjectivity), VADER (lexicon), TF-IDF + Logistic Regression / Naive Bayes / Linear SVM.
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, nltk
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import movie_reviews
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from common import *

banner("EXPERIMENT 7 - Sentiment Analysis")
here = os.path.dirname(os.path.abspath(__file__)); OUT = out_dir(__file__)
for r in ["movie_reviews", "vader_lexicon", "punkt", "punkt_tab"]: nltk.download(r, quiet=True)

csvp = os.path.join(here, "IMDB Dataset.csv")
if os.path.exists(csvp):
    df = pd.read_csv(csvp).sample(4000, random_state=1).rename(columns={"review": "text", "sentiment": "label"})
    df["label"] = df.label.map({"positive": 1, "negative": 0})
else:
    df = pd.DataFrame([(movie_reviews.raw(f), int(c == "pos")) for c in movie_reviews.categories() for f in movie_reviews.fileids(c)], columns=["text", "label"])
df["text"] = df.text.str.replace(r"<[^>]+>", " ", regex=True)
say("reviews:", len(df), "| positive:", df.label.sum(), "| negative:", (1 - df.label).sum())

# ---- Part 1: sentiment & subjectivity on sentences ----
section("Part 1: polarity and subjectivity of sample sentences")
samples = ["I absolutely love this movie, the acting is brilliant!", "This was the worst film I have ever seen.",
           "The movie was released in 2019 and runs for two hours.", "I think the plot is quite interesting but a bit slow.",
           "It is not good at all.", "The soundtrack is okay, nothing special."]
sia = SentimentIntensityAnalyzer(); rows = []
for s in samples:
    b = TextBlob(s).sentiment; v = sia.polarity_scores(s)["compound"]
    lab = "Positive" if b.polarity > .05 else "Negative" if b.polarity < -.05 else "Neutral"
    rows.append((s, round(b.polarity, 3), round(b.subjectivity, 3), round(v, 3), lab))
    say(f"{lab:8s} pol={b.polarity:+.2f} subj={b.subjectivity:.2f} vader={v:+.2f} | {s}")
sdf = pd.DataFrame(rows, columns=["sentence", "polarity", "subjectivity", "vader", "label"]); sdf.to_csv(os.path.join(OUT, "sample_sentiment.csv"), index=False)

fig, ax = plt.subplots(figsize=(8, 6.2))
col = {"Positive": PAL[2], "Negative": PAL[3], "Neutral": PAL[7]}
for _, r in sdf.iterrows():
    ax.scatter(r.polarity, r.subjectivity, s=140, color=col[r.label], edgecolor="k", zorder=3)
    ax.annotate(r.sentence[:28] + "...", (r.polarity, r.subjectivity), xytext=(6, 6), textcoords="offset points", fontsize=8)
ax.axvline(0, c="#888", lw=1); ax.set(xlim=(-1.1, 1.3), ylim=(-.05, 1.1), title="Polarity vs Subjectivity", xlabel="polarity (negative <-> positive)", ylabel="subjectivity (objective <-> subjective)")
save(fig, OUT, "polarity_subjectivity_samples.png", "Fig 1: Sample sentences on the polarity/subjectivity plane")

# ---- Part 2a: lexicon-based on full dataset ----
section("Part 2a: lexicon methods on all reviews")
df["tb_pol"] = df.text.map(lambda t: TextBlob(t).sentiment.polarity)
df["tb_subj"] = df.text.map(lambda t: TextBlob(t).sentiment.subjectivity)
df["vader"] = df.text.map(lambda t: sia.polarity_scores(t)["compound"])
for n, c in [("TextBlob", "tb_pol"), ("VADER", "vader")]:
    say(f"{n:9s} accuracy (threshold 0): {accuracy_score(df.label, (df[c] > 0).astype(int))*100:.2f}%")
say(f"mean subjectivity: {df.tb_subj.mean():.3f} | pos reviews pol {df[df.label==1].tb_pol.mean():+.3f} | neg reviews pol {df[df.label==0].tb_pol.mean():+.3f}")
df.drop(columns="text").to_csv(os.path.join(OUT, "lexicon_scores.csv"), index=False)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
for l, n, c in [(1, "positive", PAL[2]), (0, "negative", PAL[3])]:
    axs[0].hist(df[df.label == l].tb_pol, bins=35, alpha=.65, label=n, color=c); axs[1].hist(df[df.label == l].vader, bins=35, alpha=.65, label=n, color=c)
axs[0].set(title="TextBlob Polarity by True Label", xlabel="polarity"); axs[1].set(title="VADER Compound by True Label", xlabel="compound"); axs[0].legend()
save(fig, OUT, "lexicon_score_distribution.png", "Fig 2: Lexicon scores separate positive from negative reviews")

# ---- Part 2b: supervised models ----
section("Part 2b: machine-learning classifiers (TF-IDF)")
Xtr, Xte, ytr, yte = train_test_split(df.text, df.label, test_size=.25, stratify=df.label, random_state=500119453)
vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_df=.9, sublinear_tf=True)
A, B = vec.fit_transform(Xtr), vec.transform(Xte); say("TF-IDF features:", A.shape[1])
models = {"Logistic Regression": LogisticRegression(max_iter=2000, C=5), "Naive Bayes": MultinomialNB(alpha=.3), "Linear SVM": LinearSVC(C=.6)}
res = []; preds = {}; scores = {}
for n, m in models.items():
    m.fit(A, ytr); p = m.predict(B); preds[n] = p
    scores[n] = m.decision_function(B) if hasattr(m, "decision_function") else m.predict_proba(B)[:, 1]
    pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="binary")
    res.append((n, accuracy_score(yte, p), pr, rc, f1)); say(f"{n:20s} acc={res[-1][1]*100:.2f}% P={pr:.3f} R={rc:.3f} F1={f1:.3f}")
res.append(("VADER (no training)", accuracy_score(yte, (df.loc[yte.index, 'vader'] > 0).astype(int)), *precision_recall_fscore_support(yte, (df.loc[yte.index, 'vader'] > 0).astype(int), average="binary")[:3]))
rdf = pd.DataFrame(res, columns=["model", "accuracy", "precision", "recall", "f1"]); rdf.to_csv(os.path.join(OUT, "model_results.csv"), index=False)

best = rdf.iloc[:3].sort_values("accuracy").iloc[-1].model; say("best model:", best)
lr = models["Logistic Regression"]; names = np.array(vec.get_feature_names_out()); o = np.argsort(lr.coef_[0])
say("most NEGATIVE words:", list(names[o[:12]])); say("most POSITIVE words:", list(names[o[-12:]][::-1]))
for s in ["a wonderful, moving and brilliant film", "boring, predictable and a complete waste of time"]:
    say(f"predict '{s}' ->", "POSITIVE" if lr.predict(vec.transform([s]))[0] else "NEGATIVE")

# ---- figures ----
fig, ax = plt.subplots(figsize=(10, 5.2)); w = .2; x = np.arange(len(rdf))
for i, (c, l) in enumerate([("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]): ax.bar(x + i * w, rdf[c], w, label=l, color=PAL[i])
ax.set_xticks(x + 1.5 * w, rdf.model, rotation=10); ax.set(ylim=(.5, 1), title="Classifier Comparison", ylabel="score"); ax.legend(ncol=4, loc="upper left")
save(fig, OUT, "model_comparison.png", "Fig 3: Sentiment models on the IMDB test split")

fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (n, p) in zip(axs, preds.items()):
    cm = confusion_matrix(yte, p); ax.imshow(cm, cmap="Blues"); ax.grid(False)
    for (i, j), v in np.ndenumerate(cm): ax.text(j, i, v, ha="center", va="center", fontsize=14, color="w" if v > cm.max() / 2 else "k")
    ax.set_xticks([0, 1], ["neg", "pos"]); ax.set_yticks([0, 1], ["neg", "pos"]); ax.set(title=n, xlabel="predicted", ylabel="actual")
save(fig, OUT, "confusion_matrices.png", "Fig 4: Confusion matrices")

fig, ax = plt.subplots(figsize=(6.5, 5.5))
for (n, sc), c in zip(scores.items(), PAL):
    f, t, _ = roc_curve(yte, sc); ax.plot(f, t, color=c, label=f"{n} (AUC {auc(f, t):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set(title="ROC Curves", xlabel="false positive rate", ylabel="true positive rate"); ax.legend(loc="lower right")
save(fig, OUT, "roc_curves.png", "Fig 5: ROC curves of the classifiers")

fig, ax = plt.subplots(figsize=(9, 6)); idx = np.r_[o[:12], o[-12:]]
ax.barh(names[idx], lr.coef_[0][idx], color=[PAL[3] if v < 0 else PAL[2] for v in lr.coef_[0][idx]]); ax.set(title="Most Influential Words (Logistic Regression)", xlabel="coefficient")
save(fig, OUT, "top_features.png", "Fig 6: Words driving positive / negative predictions")
say("EXPERIMENT 7 COMPLETED")
