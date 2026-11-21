"""EXPERIMENT 6: N-gram language models (uni/bi/tri-gram) with smoothing, perplexity, text generation.
Author: aditya khatri 500119453
Smoothing implemented from scratch: MLE, Laplace (add-1), Add-k, Linear interpolation,
Absolute discounting / interpolated Kneser-Ney (bigram), Stupid Backoff (trigram).
Corpus: Brown 'news' category (NLTK) - works offline.
"""
import os, sys, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, nltk
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from nltk.corpus import brown
from common import *

banner("EXPERIMENT 6 - N-gram Language Model and Smoothing")
OUT = out_dir(__file__); random.seed(500119453)
nltk.download("brown", quiet=True)

# ---- data ----
sents = [[w.lower() for w in s if w.isalpha()] for s in brown.sents(categories="news")]
sents = [s for s in sents if len(s) > 3]; random.shuffle(sents)
cut = int(.9 * len(sents)); train, test = sents[:cut], sents[cut:]
cnt_all = Counter(w for s in train for w in s)
vocab = {w for w, c in cnt_all.items() if c >= 2} | {"<unk>"}
def unk(s): return [w if w in vocab else "<unk>" for w in s]
train = [["<s>", "<s>"] + unk(s) + ["</s>"] for s in train]; test = [["<s>", "<s>"] + unk(s) + ["</s>"] for s in test]
V = len(vocab) + 2
say(f"train sentences: {len(train)} | test sentences: {len(test)} | vocabulary |V| = {V}")

uni, bi, tri = Counter(), Counter(), Counter()
for s in train:
    for i, w in enumerate(s):
        uni[w] += 1
        if i >= 1: bi[(s[i - 1], w)] += 1
        if i >= 2: tri[(s[i - 2], s[i - 1], w)] += 1
N = sum(uni.values()); ctx1 = Counter(); ctx2 = Counter()
for (a, b), c in bi.items(): ctx1[a] += c
for (a, b, c3), c in tri.items(): ctx2[(a, b)] += c
cont = Counter(b for (a, b) in bi)           # continuation counts for Kneser-Ney
followers = defaultdict(int)
for (a, b) in bi: followers[a] += 1
n_bitypes = len(bi)
say(f"tokens N = {N:,} | unique bigrams = {len(bi):,} | unique trigrams = {len(tri):,}")

# ---- probability estimators ----
def p_uni(w): return uni[w] / N if uni[w] else 1e-12
def p_uni_lap(w): return (uni[w] + 1) / (N + V)
def p_bi_mle(w, a): return bi[(a, w)] / ctx1[a] if ctx1[a] else 0.0
def p_bi_lap(w, a): return (bi[(a, w)] + 1) / (ctx1[a] + V)
def p_bi_addk(w, a, k=.05): return (bi[(a, w)] + k) / (ctx1[a] + k * V)
def p_bi_interp(w, a, l=(.7, .3)): return l[0] * p_bi_mle(w, a) + l[1] * p_uni_lap(w)
def p_bi_kn(w, a, d=.75):
    pc = (cont[w] + .01) / (n_bitypes + .01 * V)
    if not ctx1[a]: return pc
    return max(bi[(a, w)] - d, 0) / ctx1[a] + d * followers[a] / ctx1[a] * pc
def p_tri_mle(w, a, b): return tri[(a, b, w)] / ctx2[(a, b)] if ctx2[(a, b)] else 0.0
def p_tri_lap(w, a, b): return (tri[(a, b, w)] + 1) / (ctx2[(a, b)] + V)
def p_tri_interp(w, a, b, l=(.6, .3, .1)): return l[0] * p_tri_mle(w, a, b) + l[1] * p_bi_kn(w, b) + l[2] * p_uni_lap(w)
def p_tri_sb(w, a, b, al=.4):   # stupid backoff (scores, not strictly normalised)
    if tri[(a, b, w)]: return tri[(a, b, w)] / ctx2[(a, b)]
    if bi[(b, w)]: return al * bi[(b, w)] / ctx1[b]
    return al * al * p_uni_lap(w)

def perplexity(data, order, f):
    ll = n = 0
    for s in data:
        for i in range(2, len(s)):
            p = f(s[i]) if order == 1 else f(s[i], s[i - 1]) if order == 2 else f(s[i], s[i - 2], s[i - 1])
            ll += math.log(max(p, 1e-12)); n += 1
    return math.exp(-ll / n)

# ---- worked example ----
section("Worked example: P(w | previous) for the bigram 'of the'")
for name, f in [("MLE", lambda: p_bi_mle("the", "of")), ("Laplace", lambda: p_bi_lap("the", "of")),
                ("Add-k(0.05)", lambda: p_bi_addk("the", "of")), ("Interpolated", lambda: p_bi_interp("the", "of")),
                ("Kneser-Ney", lambda: p_bi_kn("the", "of"))]: say(f"{name:14s} P(the | of) = {f():.5f}")
say("unseen bigram P(zebra | of): MLE =", p_bi_mle("zebra", "of"), "| Laplace =", f"{p_bi_lap('zebra','of'):.2e}", "| KN =", f"{p_bi_kn('zebra','of'):.2e}")
say("top bigrams:", [(" ".join(k), v) for k, v in bi.most_common(6)])

# ---- perplexity table ----
section("Perplexity on held-out test set (lower is better)")
models = [("Unigram (Laplace)", 1, p_uni_lap), ("Bigram MLE", 2, p_bi_mle), ("Bigram Laplace", 2, p_bi_lap),
          ("Bigram Add-k 0.05", 2, p_bi_addk), ("Bigram Interpolation", 2, p_bi_interp), ("Bigram Kneser-Ney", 2, p_bi_kn),
          ("Trigram Laplace", 3, p_tri_lap), ("Trigram Interp+KN", 3, p_tri_interp), ("Trigram Stupid Backoff", 3, p_tri_sb)]
res = [(n, perplexity(test, o, f)) for n, o, f in models]
rdf = pd.DataFrame(res, columns=["Model", "Perplexity"]); rdf.to_csv(os.path.join(OUT, "perplexity.csv"), index=False)
for n, p in res: say(f"{n:24s} perplexity = {p:12.2f}" + ("   (zero-probability clipped at 1e-12)" if n == "Bigram MLE" else ""))

# ---- tune add-k and discount ----
ks = [.001, .005, .01, .05, .1, .3, .5, 1.0]; pk = [perplexity(test, 2, lambda w, a, k=k: p_bi_addk(w, a, k)) for k in ks]
ds = [.1, .3, .5, .7, .75, .9, .95]; pd_ = [perplexity(test, 2, lambda w, a, d=d: p_bi_kn(w, a, d)) for d in ds]
say("best add-k:", ks[int(np.argmin(pk))], f"-> PP {min(pk):.1f} | best KN discount:", ds[int(np.argmin(pd_))], f"-> PP {min(pd_):.1f}")

# ---- generation ----
def generate(order=2, f=None, maxlen=18):
    s = ["<s>", "<s>"]
    for _ in range(maxlen):
        a, b = s[-2], s[-1]
        cands = [w for w in vocab if w != "<unk>"] + ["</s>"]
        if order == 2:
            pool = [(w, c) for (x, w), c in bi.items() if x == b]
        else:
            pool = [(w, c) for (x, y, w), c in tri.items() if x == a and y == b] or [(w, c) for (x, w), c in bi.items() if x == b]
        pool = [(w, c) for w, c in pool if w not in ("<s>", "<unk>")]
        if not pool: break
        ws, cs = zip(*pool); w = random.choices(ws, weights=cs)[0]
        if w == "</s>": break
        s.append(w)
    return " ".join(s[2:])
section("Generated sentences")
for o in (2, 3):
    for _ in range(3): say(f"{o}-gram:", generate(o))

# ---- figures ----
fig, ax = plt.subplots(figsize=(10, 5.5)); r = rdf[rdf.Perplexity < 1e5].sort_values("Perplexity")
bars = ax.barh(r.Model, r.Perplexity, color=[PAL[2] if i == 0 else PAL[0] for i in range(len(r))])
for b in bars: ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {b.get_width():,.0f}", va="center")
ax.invert_yaxis(); ax.set(title="Perplexity of Smoothing Methods (lower = better)", xlabel="perplexity")
save(fig, OUT, "perplexity_comparison.png", "Fig 1: Smoothing methods compared on the test set")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
axs[0].semilogx(ks, pk, "o-", color=PAL[1]); axs[0].set(title="Add-k Smoothing: choosing k", xlabel="k", ylabel="perplexity")
axs[1].plot(ds, pd_, "o-", color=PAL[4]); axs[1].set(title="Kneser-Ney: choosing discount d", xlabel="d", ylabel="perplexity")
save(fig, OUT, "hyperparameter_tuning.png", "Fig 2: Tuning smoothing parameters")

fig, ax = plt.subplots(figsize=(8, 5.5)); topw = [w for w, _ in uni.most_common(15) if w not in ("<s>", "</s>", "<unk>")][:10]
M = np.array([[p_bi_kn(w, a) for w in topw] for a in topw]); im = ax.imshow(M, cmap="YlGnBu"); ax.grid(False)
ax.set_xticks(range(10), topw, rotation=45); ax.set_yticks(range(10), topw); ax.set(title="Bigram P(next | previous), Kneser-Ney", xlabel="next word", ylabel="previous word"); fig.colorbar(im)
save(fig, OUT, "bigram_heatmap.png", "Fig 3: Smoothed bigram probabilities of frequent words")

fig, ax = plt.subplots(figsize=(8, 4.8)); fr = np.array(sorted(uni.values(), reverse=True)); ax.loglog(range(1, len(fr) + 1), fr, color=PAL[3])
ax.set(title="Zipf's Law in the Training Corpus", xlabel="rank", ylabel="frequency")
save(fig, OUT, "zipf_law.png", "Fig 4: Rank-frequency distribution (log-log)")

fig, ax = plt.subplots(figsize=(8, 4.8)); cc = Counter(bi.values()); xs = range(1, 9)
ax.bar(xs, [cc[x] for x in xs], color=PAL[6]); ax.set(title="Bigram Frequency of Frequencies", xlabel="count c", ylabel="number of bigrams seen c times")
ax.text(.5, .75, f"{cc[1]/len(bi)*100:.0f}% of bigrams occur once\n-> unseen events matter!", transform=ax.transAxes, ha="center")
save(fig, OUT, "frequency_of_frequencies.png", "Fig 5: Why smoothing is needed (sparsity)")
say("EXPERIMENT 6 COMPLETED")
