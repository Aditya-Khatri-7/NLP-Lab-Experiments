"""EXPERIMENT 5: POS tagging, tweet categorisation, chunking, and an HMM-Viterbi tagger.
Author: aditya khatri 500119453
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, nltk
import matplotlib.pyplot as plt
from collections import Counter
from nltk import pos_tag, RegexpParser
from nltk.corpus import twitter_samples, treebank
from nltk.tag import hmm
from nltk.tokenize import TweetTokenizer, word_tokenize
from common import *

banner("EXPERIMENT 5 - POS Tagging and Chunking")
OUT = out_dir(__file__)
for r in ["punkt", "punkt_tab", "averaged_perceptron_tagger_eng", "twitter_samples", "treebank", "universal_tagset"]:
    nltk.download(r, quiet=True)
tk = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=False)

# ---- Part 1: categorise & tag words in Twitter data ----
section("Part 1: tagging 500 tweets")
tweets = twitter_samples.strings("positive_tweets.json")[:250] + twitter_samples.strings("negative_tweets.json")[:250]
def prep(t): return re.sub(r"https?://\S+", "", t)
EMO = re.compile(r"^[:;=8xX][\-o^']?[)(DPp/\|3]+$|^<3$")
SLANG_PRP = {"i", "u", "im", "i'm", "ur", "ya", "y'all"}
def tag_tweet(t):
    """NLTK tagger + Twitter-specific corrections (emoticon/hashtag/mention/slang)"""
    out = []
    for w, g in pos_tag(tk.tokenize(prep(t))):
        if EMO.match(w): g = "EMO"
        elif w.startswith("#"): g = "HT"
        elif w.startswith("@"): g = "USR"
        elif w.lower() in SLANG_PRP: g = "PRP"
        out.append((w, g))
    return out
tagged_all = [tag_tweet(t) for t in tweets]
flat = [x for s in tagged_all for x in s]
say("tweets:", len(tweets), "| tokens:", len(flat))
for s in tagged_all[:3]: say(s)

def category(w, tag):
    if w.startswith("#"): return "Hashtag"
    if w.startswith("@"): return "Mention"
    if re.fullmatch(r"[:;=8][\-o^']?[)(DPp/\\|]", w) or w in (":)", ":(", ":D"): return "Emoticon"
    if tag.startswith("NN"): return "Noun"
    if tag.startswith("VB") or tag == "MD": return "Verb"
    if tag.startswith("JJ"): return "Adjective"
    if tag.startswith("RB"): return "Adverb"
    if tag in ("PRP", "PRP$", "WP", "WP$"): return "Pronoun"
    if tag in ("DT", "IN", "CC", "TO", "WDT", "WRB", "EX", "PDT", "RP"): return "Function word"
    if tag == "CD": return "Number"
    if tag == "UH": return "Interjection"
    return "Punct/Other"
cat = Counter(category(w, t) for w, t in flat)
tagc = Counter(t for _, t in flat)
say("categories:", dict(cat.most_common()))
say("top 10 tags:", tagc.most_common(10))
pd.DataFrame(flat, columns=["token", "tag"]).assign(category=[category(w, t) for w, t in flat]).to_csv(os.path.join(OUT, "tagged_tweets.csv"), index=False)
byc = {}
for w, t in flat:
    byc.setdefault(category(w, t), Counter())[w.lower()] += 1
for c in ["Noun", "Verb", "Adjective", "Hashtag", "Mention"]:
    if c in byc: say(f"{c:10s} top:", [w for w, _ in byc[c].most_common(8)])

fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
tt = tagc.most_common(15); axs[0].bar([a for a, _ in tt], [b for _, b in tt], color=PAL[0]); axs[0].set(title="Top 15 POS Tags", ylabel="count")
plt.setp(axs[0].get_xticklabels(), rotation=45)
cc = cat.most_common(); axs[1].pie([b for _, b in cc], labels=[a for a, _ in cc], autopct="%1.0f%%", colors=(PAL * 3)[:len(cc)],
                                    wedgeprops=dict(edgecolor="white"), textprops=dict(fontsize=8)); axs[1].set_title("Word Categories in Tweets")
save(fig, OUT, "pos_tag_distribution.png", "Fig 1: POS tags and word categories of Twitter data")

fig, ax = plt.subplots(figsize=(11, 5))
for i, c in enumerate(["Noun", "Verb", "Adjective", "Adverb"]):
    if c in byc:
        w, n = zip(*byc[c].most_common(6)); ax.bar(np.arange(6) + i * 7, n, color=PAL[i], label=c)
        for j, (a, b) in enumerate(zip(w, n)): ax.text(j + i * 7, b, a, rotation=60, fontsize=8, ha="left", va="bottom")
ax.set_xticks([]); ax.set(title="Most Common Words per Open Class", ylabel="count"); ax.legend(); ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
save(fig, OUT, "top_words_per_class.png", "Fig 2: Frequent nouns, verbs, adjectives, adverbs")

# ---- Part 2: POS tagging + chunking on sentences ----
section("Part 2: POS tagging and chunking")
grammar = r"""
NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}
NP: {<HT|USR>}
VP: {<MD>?<VB.*>+<RB.*>*}
PP: {<IN|TO><NP>}
"""
cp = RegexpParser(grammar)
sents = ["The intelligent student solved the difficult NLP problem in the laboratory.",
         "Machine learning models can quickly process large text corpora.",
         "She is reading a new book about natural language processing."]
rows = []
for s in sents:
    tg = pos_tag(word_tokenize(s)); tree = cp.parse(tg)
    say("sentence:", s); say("tags    :", tg); say("tree    :", str(tree).replace("\n", " "))
    for st in tree.subtrees(lambda t: t.label() in ("NP", "VP", "PP")):
        rows.append((s[:40] + "...", st.label(), " ".join(w for w, _ in st.leaves())))
crow = pd.DataFrame(rows, columns=["sentence", "chunk", "text"]); print(crow.to_string(index=False))
crow.to_csv(os.path.join(OUT, "chunks.csv"), index=False)

# tweets chunking: noun phrases
nps = Counter()
for s in tagged_all[:300]:
    for st in cp.parse(s).subtrees(lambda t: t.label() == "NP"): nps[" ".join(w.lower() for w, _ in st.leaves())] += 1
say("top noun phrases in tweets:", nps.most_common(10))

def draw_chunks(ax, tagged, tree):
    ax.axis("off"); ax.grid(False); x = 0; colors = {"NP": PAL[0], "VP": PAL[1], "PP": PAL[2]}
    words = []
    for st in tree:
        if hasattr(st, "label"): words += [(w, t, st.label()) for w, t in st.leaves()]
        else: words.append((st[0], st[1], None))
    for w, t, c in words:
        wd = .1 * max(len(w), 3) * .7 + .15
        ax.add_patch(plt.Rectangle((x, 0), wd, .6, fc=colors.get(c, "#eee"), alpha=.6 if c else 1, ec="white"))
        ax.text(x + wd / 2, .38, w, ha="center", fontsize=9); ax.text(x + wd / 2, .15, t, ha="center", fontsize=8, color="#333")
        x += wd
    ax.set_xlim(0, x); ax.set_ylim(-.2, .9)
fig, axs = plt.subplots(len(sents), 1, figsize=(13, 2.2 * len(sents)))
for ax, s in zip(axs, sents): tg = pos_tag(word_tokenize(s)); draw_chunks(ax, tg, cp.parse(tg))
axs[0].set_title("Chunked Sentences  (blue = NP, orange = VP, green = PP)")
save(fig, OUT, "chunk_diagram.png", "Fig 3: POS tags and chunks of sample sentences")

# ---- Part 3: HMM + Viterbi tagger (syllabus: POS tagging using HMM) ----
section("Part 3: HMM tagger with Viterbi decoding (Penn Treebank sample)")
data = [[(w.lower(), t) for w, t in s] for s in treebank.tagged_sents(tagset="universal")]
cut = int(len(data) * .8); train, test = data[:cut], data[cut:]
trainer = hmm.HiddenMarkovModelTrainer()
model = trainer.train_supervised(train, estimator=lambda fd, bins: nltk.LidstoneProbDist(fd, 0.01, bins))
correct = total = 0; conf = Counter()
for s in test[:300]:
    pred = model.tag([w for w, _ in s])
    for (w, g), (_, p) in zip(s, pred):
        total += 1; correct += g == p; conf[(g, p)] += 1
acc = correct / total; say(f"HMM-Viterbi accuracy on {total} test tokens: {acc*100:.2f}%")
ex = "the quick brown fox jumps over the lazy dog".split(); say("example:", model.tag(ex))
tags = sorted({g for g, _ in conf} | {p for _, p in conf})
M = np.array([[conf[(g, p)] for p in tags] for g in tags], float); M = M / M.sum(1, keepdims=True).clip(1)
fig, ax = plt.subplots(figsize=(8, 6.5)); im = ax.imshow(M, cmap="Blues"); ax.grid(False)
ax.set_xticks(range(len(tags)), tags, rotation=45); ax.set_yticks(range(len(tags)), tags)
ax.set(title=f"HMM-Viterbi Confusion Matrix (accuracy {acc*100:.1f}%)", xlabel="predicted", ylabel="gold"); fig.colorbar(im)
save(fig, OUT, "hmm_confusion.png", "Fig 4: HMM tagger row-normalised confusion matrix")
say("EXPERIMENT 5 COMPLETED")
