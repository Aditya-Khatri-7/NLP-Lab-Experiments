"""EXPERIMENT 2: Text retrieval (first 100 tweets -> file) + word & sentence tokenization.
Author: aditya khatri 500119453

Twitter API needs paid credentials, so tweets are taken from NLTK's real
`twitter_samples` corpus. An optional tweepy function is included below.
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, nltk
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import twitter_samples
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer, WhitespaceTokenizer
from common import *

banner("EXPERIMENT 2 - Text Retrieval & Tokenization")
here = os.path.dirname(os.path.abspath(__file__)); OUT = out_dir(__file__)
for r in ["punkt", "punkt_tab", "twitter_samples"]: nltk.download(r, quiet=True)

def fetch_with_tweepy(bearer_token, username, n=100):  # optional, needs credentials
    import tweepy
    c = tweepy.Client(bearer_token=bearer_token)
    uid = c.get_user(username=username).data.id
    return [t.text for t in c.get_users_tweets(uid, max_results=n).data]

# ---- Part 1: retrieve first 100 tweets and store in file ----
section("Part 1: retrieving first 100 tweets")
raw = twitter_samples.strings("tweets.20150430-223406.json")
first100 = raw[:100]
pd.DataFrame({"tweet_no": range(1, 101), "text": first100}).to_csv(os.path.join(OUT, "first_100_tweets.csv"), index=False)
with open(os.path.join(OUT, "first_100_tweets.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(t.replace("\n", " ") for t in first100))
say("total tweets available:", len(raw), "| saved 100 to first_100_tweets.csv / .txt")
for i, t in enumerate(first100[:3], 1): say(f"tweet {i}:", t.replace("\n", " ")[:110])

# ---- cleaning ----
def clean(t):
    t = re.sub(r"https?://\S+|www\.\S+", "", t)        # URLs
    t = re.sub(r"RT\s+@\w+:?", "", t)                    # retweet marks
    return re.sub(r"\s+", " ", t).strip()
cleaned = [clean(t) for t in first100]

# ---- Part 2: sentence tokenization ----
section("Part 2a: sentence tokenization")
sent_rows = []
for i, t in enumerate(cleaned, 1):
    for s in sent_tokenize(t): sent_rows.append((i, s))
sdf = pd.DataFrame(sent_rows, columns=["tweet_no", "sentence"]); sdf.to_csv(os.path.join(OUT, "sentence_tokens.csv"), index=False)
say("tweets:", len(cleaned), "-> sentences:", len(sdf), f"(avg {len(sdf)/len(cleaned):.2f} per tweet)")
demo = "Dr. Smith loves NLP! Does he teach at U.P.E.S. too? Yes, since 2020."
say("demo text:", demo); [say("  sentence:", s) for s in sent_tokenize(demo)]

# ---- word tokenization: compare tokenizers ----
section("Part 2b: word tokenization (3 tokenizers compared)")
sample = next((t for t in cleaned if re.search(r"[@#]\w+", t) and len(t) > 60), cleaned[1])
say("sample:", sample)
tw = TweetTokenizer(preserve_case=False, strip_handles=False, reduce_len=True)
say("word_tokenize :", word_tokenize(sample))
say("TweetTokenizer:", tw.tokenize(sample))
say("Whitespace    :", WhitespaceTokenizer().tokenize(sample))

wt = [tw.tokenize(t) for t in cleaned]
words = [w for ts in wt for w in ts if re.match(r"^\w+$", w)]
pd.DataFrame({"tweet_no": range(1, 101), "tokens": [" | ".join(x) for x in wt]}).to_csv(os.path.join(OUT, "word_tokens.csv"), index=False)
say("total tokens:", sum(map(len, wt)), "| word tokens:", len(words), "| vocabulary:", len(set(words)))
top = Counter(words).most_common(15); say("top 15:", top)

# ---- figures ----
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh([w for w, _ in top][::-1], [c for _, c in top][::-1], color=PAL[0])
ax.set(title="Most Frequent Tokens in First 100 Tweets", xlabel="Frequency")
save(fig, OUT, "top_tokens.png", "Fig 1: Top 15 word tokens")

lens = [len(x) for x in wt]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
axs[0].hist(lens, bins=15, color=PAL[1], edgecolor="white"); axs[0].set(title="Tokens per Tweet", xlabel="tokens", ylabel="tweets")
axs[1].hist([len(w) for w in words], bins=range(1, 16), color=PAL[2], edgecolor="white"); axs[1].set(title="Token Length", xlabel="characters")
save(fig, OUT, "token_distributions.png", "Fig 2: Tokenization statistics")

fig, ax = plt.subplots(figsize=(10, 4.2)); ax.axis("off"); ax.grid(False)
rows = [["word_tokenize", " ".join(word_tokenize(sample))[:70]], ["TweetTokenizer", " ".join(tw.tokenize(sample))[:70]],
        ["Whitespace", " ".join(WhitespaceTokenizer().tokenize(sample))[:70]]]
t = ax.table(cellText=rows, colLabels=["Tokenizer", "Output (first 70 chars)"], loc="center", cellLoc="left", colWidths=[.22, .78])
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 2.2)
ax.set_title("Tokenizer Comparison on One Tweet")
save(fig, OUT, "tokenizer_comparison.png", "Fig 3: How tokenizers treat hashtags/mentions/emoticons")
say("EXPERIMENT 2 COMPLETED")
