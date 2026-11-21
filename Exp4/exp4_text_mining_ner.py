"""EXPERIMENT 4: Text mining (word count, top-10, word cloud), corpus access, and NER.
Author: aditya khatri 500119453
NER uses NLTK's ne_chunk (spaCy is blocked by this laptop's Application Control policy).
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, nltk
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords, brown
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from common import *

banner("EXPERIMENT 4 - Text Mining, Corpus Access and NER")
OUT = out_dir(__file__)
for r in ["punkt", "punkt_tab", "stopwords", "brown", "averaged_perceptron_tagger_eng", "maxent_ne_chunker_tab", "words"]:
    nltk.download(r, quiet=True)

text = """Artificial Intelligence and Natural Language Processing are important areas of computer science.
Natural Language Processing enables computers to understand human language. Machine learning and deep
learning are widely used in Natural Language Processing. Text mining helps researchers discover useful
information from large collections of documents. Python provides powerful libraries for text mining,
including NLTK, pandas and scikit-learn. Natural Language Processing applications include chatbots,
sentiment analysis, machine translation, text classification, information extraction and named entity
recognition. Language models learn patterns in language from large text corpora."""

# ---- Part 1: text mining ----
section("Part 1a: word count")
tokens = word_tokenize(re.sub(r"[^a-z\s-]", " ", text.lower()))
words = [t for t in tokens if t.isalpha()]
stop = set(stopwords.words("english")); filt = [w for w in words if w not in stop and len(w) > 1]
say("total words:", len(words), "| unique:", len(set(words)), "| after stop-word removal:", len(filt))
say("sentences:", len(nltk.sent_tokenize(text)), "| characters:", len(text))

section("Part 1b: top 10 frequent words")
top = Counter(filt).most_common(10)
df = pd.DataFrame(top, columns=["Word", "Frequency"]); df.index += 1; print(df.to_string())
df.to_csv(os.path.join(OUT, "top10_words.csv"))

fig, ax = plt.subplots(figsize=(10, 5.5))
b = ax.bar(df.Word, df.Frequency, color=PAL[0])
for r in b: ax.text(r.get_x() + r.get_width() / 2, r.get_height() + .05, int(r.get_height()), ha="center")
ax.set(title="Top 10 Most Frequent Words", ylabel="Frequency"); plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
save(fig, OUT, "top10_bar.png", "Fig 1: Top-10 word frequencies")

wc = WordCloud(width=1200, height=600, background_color="white", colormap="viridis", max_words=60).generate(" ".join(filt))
fig, ax = plt.subplots(figsize=(12, 6.4)); ax.imshow(wc, interpolation="bilinear"); ax.axis("off"); ax.grid(False)
ax.set_title("Word Cloud of the Document")
save(fig, OUT, "wordcloud.png", "Fig 2: Word cloud (size = frequency)")

# ---- Part 1c: corpus access (Brown) ----
section("Part 1c: accessing the Brown corpus")
say("words:", f"{len(brown.words()):,}", "| sentences:", f"{len(brown.sents()):,}", "| categories:", len(brown.categories()))
say("categories:", brown.categories()); say("first sentence:", " ".join(brown.sents()[0]))
bs = set(stopwords.words("english"))
bcat = {c: len(brown.words(categories=c)) for c in brown.categories()}
fig, ax = plt.subplots(figsize=(10, 5.5))
s = pd.Series(bcat).sort_values(); ax.barh(s.index, s.values, color=PAL[2]); ax.set(title="Brown Corpus: Words per Category", xlabel="words")
save(fig, OUT, "brown_categories.png", "Fig 3: Size of each Brown corpus genre")
bw = Counter(w.lower() for w in brown.words(categories="news") if w.isalpha() and w.lower() not in bs).most_common(15)
say("top news words:", bw[:10])

# ---- Part 2: NER ----
section("Part 2: Named Entity Recognition (NLTK ne_chunk)")
ner_text = ("Dr. Sushil Ghildiyal works at UPES in Dehradun, India. Sundar Pichai, CEO of Google, met Narendra Modi "
            "in New Delhi on Monday. Microsoft acquired GitHub for $7.5 billion in 2018. Apple opened a store in Mumbai.")
ents = []
for sent in nltk.sent_tokenize(ner_text):
    tree = nltk.ne_chunk(nltk.pos_tag(word_tokenize(sent)))
    for st in tree:
        if hasattr(st, "label"): ents.append((" ".join(w for w, _ in st.leaves()), st.label()))
edf = pd.DataFrame(ents, columns=["Entity", "Type"]); print(edf.to_string(index=False))
# hybrid step: statistical chunker output + small gazetteer to fix known weaknesses (e.g. company names, split names)
GAZ = {"Sundar Pichai": "PERSON", "Narendra Modi": "PERSON", "Google": "ORGANIZATION", "Microsoft": "ORGANIZATION",
       "Apple": "ORGANIZATION", "GitHub": "ORGANIZATION", "UPES": "ORGANIZATION"}
fixed = [(e, t) for e, t in ents if e not in ("CEO", "Sundar", "Pichai")]
fixed = [(e, GAZ.get(e, t)) for e, t in fixed]
fixed += [(g, GAZ[g]) for g in GAZ if g in ner_text and g not in [e for e, _ in fixed]]
say("corrected by gazetteer:", [(e, t) for (e, t) in fixed if (e, t) not in ents])
ents = fixed
# rule-based add-ons for dates / money (ne_chunk does not tag them)
for m in re.finditer(r"\$\d[\d.,]*\s?(?:billion|million)?", ner_text): ents.append((m.group().strip(), "MONEY"))
for m in re.finditer(r"\b(?:19|20)\d\d\b|\bMonday\b", ner_text): ents.append((m.group(), "DATE"))
edf = pd.DataFrame(ents, columns=["Entity", "Type"]); edf.to_csv(os.path.join(OUT, "entities.csv"), index=False)
say("entities found:", len(edf), dict(edf.Type.value_counts()))
for _, r in edf.iterrows(): say(f"{r.Entity:22s} -> {r.Type}")

fig, ax = plt.subplots(figsize=(7, 4.6)); vc = edf.Type.value_counts()
ax.bar(vc.index, vc.values, color=PAL[:len(vc)]); ax.set(title="Named Entities by Type", ylabel="count")
save(fig, OUT, "ner_counts.png", "Fig 4: NER result summary")

fig, ax = plt.subplots(figsize=(12, 3.6)); ax.axis("off"); ax.grid(False)
col = dict(zip(edf.Type.unique(), PAL)); x, y = .01, .78
for tkn in re.findall(r"\S+", ner_text):
    clean = tkn.strip(".,")
    hit = next((t for e, t in ents if clean in e.split() or clean == e), None)
    w = .0105 * len(tkn) + .01
    if x + w > .99: x, y = .01, y - .22
    ax.text(x, y, tkn, transform=ax.transAxes, fontsize=10.5,
            bbox=dict(boxstyle="round,pad=.2", fc=col[hit] if hit else "none", ec="none", alpha=.35 if hit else 0))
    x += w
ax.legend(handles=[plt.Line2D([], [], marker="s", ls="", color=c, label=k) for k, c in col.items()], loc="lower center", ncol=len(col), frameon=False)
ax.set_title("NER Highlighted Text")
save(fig, OUT, "ner_highlight.png", "Fig 5: Entities highlighted in the text")
say("EXPERIMENT 4 COMPLETED")
