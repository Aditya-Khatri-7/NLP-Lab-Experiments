"""Builds the lab report PDF.  Author: aditya khatri 500119453"""
import os, sys, textwrap
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, Image,
                                PageBreak, KeepTogether, Preformatted, NextPageTemplate, CondPageBreak)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
from PIL import Image as PILImage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from content import EXPS

NAME = "aditya khatri 500119453"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUTPDF = os.path.join(HERE, "NLP_Lab_Report_Aditya_Khatri_500119453.pdf")
DATE = "21 November 2026"

fd = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("Mono", os.path.join(fd, "DejaVuSansMono.ttf")))
pdfmetrics.registerFont(TTFont("MonoB", os.path.join(fd, "DejaVuSansMono-Bold.ttf")))

# ---------- styles ----------
S = lambda n, **k: ParagraphStyle(n, **k)
body = S("body", fontName="Times-Roman", fontSize=11, leading=15, alignment=TA_JUSTIFY, spaceAfter=5)
bul = S("bul", parent=body, leftIndent=16, bulletIndent=4, spaceAfter=3)
h1 = S("h1", fontName="Times-Bold", fontSize=17, leading=21, spaceBefore=6, spaceAfter=8, textColor=colors.HexColor("#14325c"))
h2 = S("h2", fontName="Times-Bold", fontSize=12.5, leading=16, spaceBefore=9, spaceAfter=4, textColor=colors.HexColor("#14325c"))
cap = S("cap", fontName="Times-Italic", fontSize=9.5, leading=12, alignment=TA_CENTER, spaceAfter=2)
capnote = S("capnote", fontName="Times-Roman", fontSize=9.5, leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#333333"), spaceAfter=10)
code = S("code", fontName="Mono", fontSize=7.2, leading=8.7)
out = S("out", fontName="Mono", fontSize=7, leading=8.5, textColor=colors.HexColor("#1b1b1b"))
c_ctr = lambda n, sz, b=False, **k: S(n, fontName="Times-Bold" if b else "Times-Roman", fontSize=sz, leading=sz * 1.3, alignment=TA_CENTER, **k)
qst = S("q", parent=body, fontName="Times-Bold", spaceAfter=1)
cell = S("cell", fontName="Times-Roman", fontSize=10, leading=13, alignment=TA_LEFT)
cellb = S("cellb", parent=cell, fontName="Times-Bold")

esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def wrap_lines(text, width):
    res = []
    for ln in text.expandtabs(4).split("\n"):
        ind = len(ln) - len(ln.lstrip())
        parts = textwrap.wrap(ln, width, subsequent_indent=" " * (ind + 4), drop_whitespace=False, break_long_words=True) or [""]
        res += parts
    return "\n".join(res)

class Doc(BaseDocTemplate):
    def __init__(self, fn):
        super().__init__(fn, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.2 * cm, bottomMargin=2.2 * cm,
                         title="NLP Lab Report - Aditya Khatri 500119453", author="Aditya Khatri")
        fr = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("cover", [fr], onPage=lambda c, d: None),
                               PageTemplate("main", [fr], onPage=self.decor)])
    def decor(self, c, d):
        c.saveState(); c.setStrokeColor(colors.HexColor("#14325c")); c.setLineWidth(.6)
        c.line(2.2 * cm, A4[1] - 1.6 * cm, A4[0] - 2.2 * cm, A4[1] - 1.6 * cm)
        c.setFont("Times-Italic", 9); c.setFillColor(colors.HexColor("#333333"))
        c.drawString(2.2 * cm, A4[1] - 1.4 * cm, "Computational Linguistics and NLP Lab (CSEG4040P)")
        c.drawRightString(A4[0] - 2.2 * cm, A4[1] - 1.4 * cm, "School of Computer Science, UPES")
        c.line(2.2 * cm, 1.6 * cm, A4[0] - 2.2 * cm, 1.6 * cm)
        c.drawString(2.2 * cm, 1.15 * cm, NAME + "  |  SAP ID R2142230710")
        c.drawRightString(A4[0] - 2.2 * cm, 1.15 * cm, f"Page {d.page}")
        c.restoreState()
    def afterFlowable(self, fl):
        if isinstance(fl, Paragraph) and fl.style.name in ("h1",):
            t = fl.getPlainText(); self.notify("TOCEntry", (0, t, self.page))

def image(path, maxw=15.2 * cm, maxh=9.6 * cm):
    w, h = PILImage.open(path).size; r = min(maxw / w, maxh / h); return Image(path, w * r, h * r)

def P(t, st=body): return Paragraph(t, st)
def bullets(items): return [Paragraph(esc(i), bul, bulletText="•") for i in items]
def numbered(items): return [Paragraph(esc(i), bul, bulletText=f"{n}.") for n, i in enumerate(items, 1)]

story = []
# ================= COVER (format of the given front page) =================
story += [Spacer(1, 1.2 * cm), P("NLP Lab Report", c_ctr("t1", 26, True)), Spacer(1, .6 * cm),
          P("Computational Linguistics and Natural Language Processing Lab", c_ctr("t2", 18, True)),
          Spacer(1, 1.0 * cm), P("For", c_ctr("t3", 12)), Spacer(1, .7 * cm),
          P("Course CSEG4040P - Experiments 1 to 10", c_ctr("t4", 13, True)),
          Spacer(1, .8 * cm), P(DATE, c_ctr("t5", 12, True)), Spacer(1, 1.0 * cm), P("Prepared by", c_ctr("t6", 12)), Spacer(1, .3 * cm)]
tb = Table([[P("Specialization", c_ctr("a", 11, True)), P("SAP ID", c_ctr("a", 11, True)), P("Name", c_ctr("a", 11, True))],
            [P("AI &amp; ML", c_ctr("b", 10)), P("R2142230710", c_ctr("b", 10)), P("Aditya Khatri", c_ctr("b", 10))]], colWidths=[4.2 * cm, 4.2 * cm, 4.2 * cm])
tb.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .8, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
story += [tb, Spacer(1, .25 * cm), P("Roll No.: 500119453", c_ctr("t7", 10.5)), Spacer(1, 1.2 * cm)]
logo = PILImage.open(os.path.join(HERE, "upes_logo.png")); lw = 6.2 * cm
story += [Image(os.path.join(HERE, "upes_logo.png"), lw, lw * logo.height / logo.width), Spacer(1, .8 * cm),
          P("AI Cluster", c_ctr("t8", 12)), P("School Of Computer Science", c_ctr("t9", 12)),
          P("UNIVERSITY OF PETROLEUM &amp; ENERGY STUDIES,", c_ctr("t10", 12)), P("DEHRADUN- 248007. Uttarakhand", c_ctr("t11", 12)),
          NextPageTemplate("main"), PageBreak()]

# ================= ABOUT / INDEX =================
story.append(P("Index", S("h1i", parent=h1)))
toc = TableOfContents(); toc.levelStyles = [S("toc0", fontName="Times-Roman", fontSize=12, leading=21, leftIndent=10, rightIndent=30)]
toc.dotsMinLevel = 0; story += [toc, PageBreak()]

story.append(P("Declaration and About this Report", h1))
story += [P("This report contains the ten experiments of the Computational Linguistics and Natural Language Processing Lab (CSEG4040P) performed by <b>Aditya Khatri</b> (SAP ID R2142230710, Roll No. 500119453), AI &amp; ML, School of Computer Science, UPES Dehradun. "
            "Every experiment was written in Python, executed on a laptop and verified; the figures, numbers and console outputs in this report are the real results of those runs."),
          P("<b>Source code:</b> https://github.com/Aditya-Khatri-7/NLP-Lab-Experiments"),
          P("<b>Every program</b> prints the line <i>aditya khatri 500119453</i> in its banner and output lines, and every figure caption carries the same line."),
          P("Structure of each experiment: Aim, Objectives, Tools and Dataset, Theory, Procedure, Source Code, Output, Figures, Observations and Result, Conclusion, Viva Questions.", body),
          Spacer(1, .3 * cm), P("Summary of the experiments", h2)]
rows = [[P("Exp", cellb), P("Title", cellb), P("Key result", cellb)],
        [P("1", cell), P("Employee CSV analysis", cell), P("Data Science has the highest average salary (97,253.53); Support the lowest (45,003.67)", cell)],
        [P("2", cell), P("Text retrieval and tokenization", cell), P("100 tweets saved; 160 sentences, 2,024 tokens, vocabulary 646", cell)],
        [P("3", cell), P("Pre-processing and morphology", cell), P("Tokens reduced by 54.5 %; lemmatization vs stemming compared", cell)],
        [P("4", cell), P("Text mining, corpus, NER", cell), P("Top-10 words, word cloud, Brown corpus (1.16 M words), 15 entities", cell)],
        [P("5", cell), P("POS tagging and chunking", cell), P("Twitter tagging; NP/VP/PP chunks; HMM-Viterbi accuracy 92.15 %", cell)],
        [P("6", cell), P("N-gram model and smoothing", cell), P("Kneser-Ney perplexity 220.18 (best); MLE 249,670", cell)],
        [P("7", cell), P("Sentiment analysis", cell), P("Logistic Regression 86.4 % vs VADER 63.5 %", cell)],
        [P("8", cell), P("Summarization and meaning", cell), P("TextRank best ROUGE-1 (0.447); Lesk WSD; WordNet similarity", cell)],
        [P("9 &amp; 10", cell), P("Mini-project", cell), P("Review intelligence system: search, sentiment (85.75 %), summary, QA, topics", cell)]]
t = Table(rows, colWidths=[1.6 * cm, 5.2 * cm, 9.6 * cm], repeatRows=1)
t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe6f5")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
story += [t, PageBreak()]

# ================= EXPERIMENTS =================
for n, e in enumerate(EXPS, 1):
    label = "Experiments 9 &amp; 10" if e["key"] == "Exp9_10" else f"Experiment {n}"
    story.append(P(f"{label}: {esc(e['title'])}", h1))
    folder = os.path.join(ROOT, e["folder"]); fig_prefix = "9-10" if e["key"] == "Exp9_10" else str(n)
    story += [P("Aim", h2), P(esc(e["aim"])), P("Objectives", h2)] + bullets(e["objectives"]) + [P("Tools and Dataset", h2), P(esc(e["tools"])), P("Theory", h2)]
    story += [P(esc(x)) for x in e["theory"]] + [P("Procedure / Algorithm", h2)] + numbered(e["steps"])
    # code
    src = open(os.path.join(folder, e["script"]), encoding="utf-8").read().rstrip()
    story += [PageBreak(), P(f"Source Code - {e['script']}", h2)]
    nums = "\n".join(f"{i:3d}| {l}" for i, l in enumerate(src.split("\n"), 1))
    story.append(Preformatted(wrap_lines(nums, 106), code, maxLineLength=None))
    # output
    co = open(os.path.join(folder, "outputs", "console_output.txt"), encoding="utf-8").read().split("\n")
    shown = [l for l in co if l.strip()][:30]
    story += [PageBreak(), P("Output (console)", h2), P("The first lines of the real console output of the program (saved in outputs/console_output.txt):", body)]
    story.append(Table([[Preformatted(wrap_lines("\n".join(shown), 98), out)]], colWidths=[16.4 * cm],
                       style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3f5f8")), ("BOX", (0, 0), (-1, -1), .5, colors.grey), ("LEFTPADDING", (0, 0), (-1, -1), 5)])))
    story += [CondPageBreak(11 * cm), Spacer(1, .4 * cm), P("Output Figures", h2)]
    for i, (fn, c1, note) in enumerate(e["figs"], 1):
        story.append(KeepTogether([image(os.path.join(folder, "outputs", fn)), Spacer(1, 2),
                                   P(f"Fig {fig_prefix}.{i}: {esc(c1)} - {NAME}", cap), P(esc(note), capnote)]))
    story += [CondPageBreak(7 * cm), P("Observations and Result", h2)] + bullets(e["obs"])
    story += [P("Conclusion", h2), P(esc(e["conclusion"])), P("Viva Questions and Answers", h2)]
    for i, (q, a) in enumerate(e["viva"], 1):
        story += [P(f"Q{i}. {esc(q)}", qst), P(f"Ans: {esc(a)}", bul)]
    story.append(PageBreak())

# ================= overall =================
story.append(P("Overall Conclusion", h1))
story += [P("In this lab the complete life cycle of a natural language processing system was implemented: data collection and cleaning (Experiments 1-3), exploration and information extraction (Experiment 4), linguistic analysis with tagging and chunking (Experiment 5), statistical language modelling with smoothing (Experiment 6), and the applications sentiment analysis (Experiment 7), summarization (Experiment 8) and an integrated mini-project (Experiments 9 and 10)."),
          P("Main learnings: (1) pre-processing choices such as stemming versus lemmatization change the quality of later steps; (2) probabilistic models such as HMMs and N-grams are strong but need smoothing for unseen events - Kneser-Ney was the best smoothing method here; (3) supervised TF-IDF models clearly outperform simple lexicon methods for sentiment; (4) extractive summarizers can be compared with proxy metrics, with TextRank being the best on overlap; (5) real applications combine many NLP components."),
          P("Limitations and future work: spaCy models could not be used on the development laptop, so NLTK was used for NER and POS tagging; the corpora are small compared with modern data sets. Future work includes word embeddings, LSTM/GRU and transformer (BERT) models, and abstractive summarization.")]
story += [P("References", h2)] + bullets([
    "D. Jurafsky and J. H. Martin, Speech and Language Processing, 2nd/3rd Edition, Pearson/Prentice Hall.",
    "C. D. Manning and H. Schutze, Foundations of Statistical Natural Language Processing, MIT Press.",
    "S. Bird, E. Klein and E. Loper, Natural Language Processing with Python (NLTK Book), O'Reilly.",
    "R. Mihalcea and P. Tarau, TextRank: Bringing Order into Texts, EMNLP 2004.",
    "R. Kneser and H. Ney, Improved backing-off for M-gram language modeling, ICASSP 1995.",
    "C. J. Hutto and E. Gilbert, VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text, ICWSM 2014.",
    "Scikit-learn, NLTK, pandas, matplotlib and wordcloud documentation.",
    "CSEG4040P Course syllabus, School of Computer Science, UPES."])

Doc(OUTPDF).multiBuild(story)
print("built", OUTPDF, os.path.getsize(OUTPDF) // 1024, "KB")
