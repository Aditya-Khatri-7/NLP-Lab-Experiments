"""EXPERIMENT 3: Text pre-processing (tokenization, stop-words, stemming, lemmatization)
and morphological analysis on movie reviews.
Author: aditya khatri 500119453

Dataset: NLTK movie_reviews (2000 IMDb reviews) - works offline.
If you have 'IMDB Dataset.csv' (Kaggle 50k) next to this file it is used instead.
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, nltk
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords, wordnet, movie_reviews
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer, LancasterStemmer
from common import *

banner("EXPERIMENT 3 - Processing Data (pre-processing + morphology)")
here = os.path.dirname(os.path.abspath(__file__)); OUT = out_dir(__file__)
for r in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger_eng", "movie_reviews"]:
    nltk.download(r, quiet=True)

# ---- load data ----
csvp = os.path.join(here, "IMDB Dataset.csv")
if os.path.exists(csvp):
    df = pd.read_csv(csvp).head(1000).rename(columns={"review": "text", "sentiment": "label"})
else:
    df = pd.DataFrame([(movie_reviews.raw(f), c) for c in movie_reviews.categories()
                       for f in movie_reviews.fileids(c)[:500]], columns=["text", "label"])
say("reviews loaded:", len(df), "| labels:", dict(df.label.value_counts()))

stop = set(stopwords.words("english")); lem = WordNetLemmatizer(); ps = PorterStemmer()
def wn_pos(tag): return {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}.get(tag[0], wordnet.NOUN)

# ---- Part 1: step by step on one review ----
section("Part 1: pipeline on a single review")
text = df.text.iloc[3][:420]
say("original:", text.replace("\n", " "))
low = re.sub(r"<[^>]+>", " ", text.lower()); say("1 lowercase + HTML removal done")
nopunct = re.sub(r"[^a-z\s]", " ", low)
toks = word_tokenize(nopunct); say("2 tokens:", len(toks), toks[:15])
nostop = [t for t in toks if t not in stop and len(t) > 2]; say("3 after stop-word removal:", len(nostop), nostop[:15])
stems = [ps.stem(t) for t in nostop]; say("4 stems:", stems[:15])
tagged = nltk.pos_tag(nostop)
lems = [lem.lemmatize(w, wn_pos(t)) for w, t in tagged]; say("5 lemmas (POS-aware):", lems[:15])
cmp = pd.DataFrame({"token": nostop, "stem": stems, "lemma": lems})
section("Stem vs lemma comparison (rows where they differ)"); print(cmp[cmp.stem != cmp.lemma].head(12).to_string(index=False))

# ---- Part 2: whole corpus ----
def preprocess(t):
    t = re.sub(r"[^a-z\s]", " ", re.sub(r"<[^>]+>", " ", t.lower()))
    k = [w for w in word_tokenize(t) if w not in stop and len(w) > 2]
    return [lem.lemmatize(w, wn_pos(g)) for w, g in nltk.pos_tag(k)]
section("Part 2: pre-processing full dataset")
df["raw_tokens"] = df.text.map(lambda t: len(word_tokenize(t)))
df["clean"] = df.text.map(preprocess); df["clean_tokens"] = df.clean.map(len)
df.assign(clean=df.clean.map(" ".join)).drop(columns="text").to_csv(os.path.join(OUT, "processed_reviews.csv"), index=False)
say(f"raw tokens: {df.raw_tokens.sum():,} -> clean tokens: {df.clean_tokens.sum():,} "
    f"({100*(1-df.clean_tokens.sum()/df.raw_tokens.sum()):.1f}% reduced)")
cnt = {l: Counter(w for c in df[df.label == l].clean for w in c) for l in df.label.unique()}
for l, c in cnt.items(): say(l, "top words:", c.most_common(8))

# ---- Part 3: Morphological analysis ----
section("Part 3: morphological analysis")
words = ["running", "happily", "unhappiness", "studies", "better", "children", "playing", "organization", "went", "mice"]
snow, lan = SnowballStemmer("english"), LancasterStemmer()
rows = []
for w in words:
    rows.append([w, ps.stem(w), snow.stem(w), lan.stem(w), lem.lemmatize(w, "v"), lem.lemmatize(w, "n"), lem.lemmatize(w, "a")])
mdf = pd.DataFrame(rows, columns=["word", "Porter", "Snowball", "Lancaster", "lemma(v)", "lemma(n)", "lemma(a)"])
print(mdf.to_string(index=False))
mdf.to_csv(os.path.join(OUT, "morphology_table.csv"), index=False)

prefixes = ["un", "re", "dis", "pre", "in", "mis"]; suffixes = ["ness", "ing", "ed", "ly", "tion", "ment", "able", "er", "s"]
def known(r):  # is r (or a spelling variant of r) a real English word?
    return any(wordnet.synsets(v) for v in {r, r + "e", r[:-1] + "y" if r.endswith("i") else r, r[:-1] if len(r) > 3 and r[-1] == r[-2] else r})
def segment(w):
    """dictionary-validated prefix + root + suffix split"""
    for p in [x for x in prefixes if w.startswith(x)] + [""]:
        rest = w[len(p):]
        for s in [x for x in suffixes if rest.endswith(x) and len(rest) > len(x)] + [""]:
            root = rest[:len(rest) - len(s)] if s else rest
            if len(root) > 2 and known(root) and (p or s): return p, root, s
    return "", w, ""
section("Affix segmentation (prefix + root + suffix)")
for w in ["unhappiness", "replaying", "disagreement", "preheated", "misunderstanding", "readable"]:
    say(w, "->", segment(w))

# ---- figures ----
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
axs[0].bar(["Raw tokens", "After clean-up"], [df.raw_tokens.sum(), df.clean_tokens.sum()], color=[PAL[7], PAL[2]])
axs[0].set(title="Token Count Before / After Pre-processing", ylabel="tokens")
for i, v in enumerate([df.raw_tokens.sum(), df.clean_tokens.sum()]): axs[0].text(i, v, f"{v:,}", ha="center", va="bottom")
for l, col in zip(df.label.unique(), PAL): axs[1].hist(df[df.label == l].clean_tokens, bins=30, alpha=.6, label=l, color=col)
axs[1].set(title="Clean Tokens per Review", xlabel="tokens", ylabel="reviews"); axs[1].legend()
save(fig, OUT, "preprocessing_effect.png", "Fig 1: Effect of pre-processing")

fig, axs = plt.subplots(1, len(cnt), figsize=(12, 5))
for ax, (l, c), col in zip(axs, cnt.items(), PAL):
    w, n = zip(*c.most_common(12)); ax.barh(w[::-1], n[::-1], color=col); ax.set_title(f"Top words: {l}")
save(fig, OUT, "top_words_by_class.png", "Fig 2: Most frequent lemmas per sentiment class")

fig, ax = plt.subplots(figsize=(11, 4.6)); ax.axis("off"); ax.grid(False)
t = ax.table(cellText=mdf.values, colLabels=mdf.columns, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.8)
for j in range(len(mdf.columns)): t[0, j].set_facecolor("#cfe0f5")
ax.set_title("Stemming vs Lemmatization (morphological analysis)")
save(fig, OUT, "morphology_table.png", "Fig 3: Stemmers and lemmatizer compared")

pos = Counter(g[:2] for _, g in nltk.pos_tag([w for c in df.clean.head(100) for w in c]))
fig, ax = plt.subplots(figsize=(8, 4.8)); ax.bar(pos.keys(), pos.values(), color=PAL[4]); ax.set(title="POS Mix of Cleaned Tokens (100 reviews)", ylabel="count")
save(fig, OUT, "pos_distribution.png", "Fig 4: Part-of-speech distribution")
say("EXPERIMENT 3 COMPLETED")
