"""Builds the editable Word (.docx) lab report.  Author: aditya khatri 500119453"""
import os, sys
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from PIL import Image as PILImage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from content import EXPS

NAME = "aditya khatri 500119453"; DATE = "21 November 2026"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "NLP_Lab_Report_Aditya_Khatri_500119453.docx")
NAVY = RGBColor(0x14, 0x32, 0x5C)

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.2); sec.top_margin = sec.bottom_margin = Cm(2.2)
sec.different_first_page_header_footer = True

# ---------- styles ----------
st = doc.styles
n = st["Normal"]; n.font.name = "Times New Roman"; n.font.size = Pt(11)
n.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
n.paragraph_format.space_after = Pt(5); n.paragraph_format.line_spacing = 1.15
for name, size, before in (("Heading 1", 17, 6), ("Heading 2", 12.5, 9)):
    h = st[name]; h.font.name = "Times New Roman"; h.font.size = Pt(size); h.font.bold = True; h.font.color.rgb = NAVY
    rf = h.element.rPr.rFonts
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None: del rf.attrib[qn(a)]
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"): rf.set(qn(a), "Times New Roman")
    h.paragraph_format.space_before = Pt(before); h.paragraph_format.space_after = Pt(4); h.paragraph_format.keep_with_next = True

def fld(par, code):
    for t, txt in (("begin", None), (None, code), ("separate", None), (None, "1"), ("end", None)):
        r = par.add_run()
        if t:
            e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t); r._r.append(e)
        elif txt == code:
            e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = code; r._r.append(e)
        else: r.text = txt
    return par

def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr(); s = OxmlElement("w:shd"); s.set(qn("w:val"), "clear"); s.set(qn("w:fill"), color); tcPr.append(s)

def para(text="", bold=False, italic=False, size=None, align=None, after=None, before=None, color=None, style=None, keep=False):
    p = doc.add_paragraph(style=style); r = p.add_run(text); r.bold = bold; r.italic = italic
    if size: r.font.size = Pt(size)
    if color: r.font.color.rgb = color
    p.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
    if after is not None: p.paragraph_format.space_after = Pt(after)
    if before is not None: p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.keep_with_next = keep
    return p

def centre(text, size=12, bold=False, after=6, before=0):
    return para(text, bold=bold, size=size, align=WD_ALIGN_PARAGRAPH.CENTER, after=after, before=before)

def item(text, mark):
    p = doc.add_paragraph(); p.paragraph_format.left_indent = Cm(0.9); p.paragraph_format.first_line_indent = Cm(-0.6)
    p.paragraph_format.space_after = Pt(3); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; p.add_run(f"{mark}\t{text}")
    p.paragraph_format.tab_stops.add_tab_stop(Cm(0.9)); return p

def mono(par_text_lines, size=7, shade_color=None):
    """one paragraph per line (Consolas), tight spacing"""
    for ln in par_text_lines:
        p = doc.add_paragraph(); pf = p.paragraph_format; pf.space_after = Pt(0); pf.space_before = Pt(0); pf.line_spacing = 1.0
        r = p.add_run(ln if ln else " "); r.font.name = "Consolas"; r.font.size = Pt(size)
        r._r.rPr.rFonts.set(qn("w:eastAsia"), "Consolas"); r._r.rPr.rFonts.set(qn("w:cs"), "Consolas")

def figure(path, cap, note):
    w, h = PILImage.open(path).size; r = min(15.2 / (w / 96 * 2.54), 9.6 / (h / 96 * 2.54)); wc = min(15.2, 9.6 * w / h)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.keep_with_next = True; p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(path, width=Cm(wc))
    c = para(cap, italic=True, size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER, after=2, keep=True)
    para(note, size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER, after=10, color=RGBColor(0x33, 0x33, 0x33))

def page_break(): doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

# ---------- header / footer (all pages except the cover) ----------
hp = sec.header.paragraphs[0]; hp.text = "Computational Linguistics and NLP Lab (CSEG4040P)\t\tSchool of Computer Science, UPES"
for r in hp.runs: r.italic = True; r.font.size = Pt(9)
fp = sec.footer.paragraphs[0]
r = fp.add_run(f"{NAME}  |  SAP ID R2142230710\t\tPage "); r.italic = True; r.font.size = Pt(9)
fld(fp, "PAGE")
for r in fp.runs: r.italic = True; r.font.size = Pt(9)

# ================= COVER (format of the given front page) =================
centre("NLP Lab Report", 26, True, after=14, before=40)
centre("Computational Linguistics and Natural Language Processing Lab", 18, True, after=26)
centre("For", 12, after=16)
centre("Course CSEG4040P - Experiments 1 to 10", 13, True, after=18)
centre(DATE, 12, True, after=24)
centre("Prepared by", 12, after=6)
t = doc.add_table(rows=2, cols=3); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
for j, (a, b) in enumerate(zip(["Specialization", "SAP ID", "Name"], ["AI & ML", "R2142230710", "Aditya Khatri"])):
    for row, txt, bold in ((0, a, True), (1, b, False)):
        c = t.cell(row, j); c.width = Cm(4.2); c.text = ""; p = c.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(txt); run.bold = bold; run.font.size = Pt(11 if bold else 10); p.paragraph_format.space_after = Pt(2)
centre("Roll No.: 500119453", 10.5, after=22, before=6)
lg = PILImage.open(os.path.join(HERE, "upes_logo.png")); p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run().add_picture(os.path.join(HERE, "upes_logo.png"), width=Cm(7.0)); p.paragraph_format.space_after = Pt(14)
for line in ["AI Cluster", "School Of Computer Science", "UNIVERSITY OF PETROLEUM & ENERGY STUDIES,", "DEHRADUN- 248007. Uttarakhand"]:
    centre(line, 12, after=0)
page_break()

# ================= INDEX (real Word table of contents) =================
doc.add_paragraph("Index", style="Heading 2").runs[0].font.size = Pt(17)
tp = doc.add_paragraph(); r = tp.add_run()
for t_, txt in (("begin", None), (None, 'TOC \\o "1-1" \\h \\z \\u'), ("separate", None), (None, "Right-click here and choose Update Field to refresh the index."), ("end", None)):
    if t_:
        e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t_); r._r.append(e)
    elif txt.startswith("TOC"):
        e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt; r._r.append(e)
    else:
        r = tp.add_run(txt); r2 = r
page_break()

# ================= ABOUT =================
doc.add_heading("Declaration and About this Report", level=1)
para("This report contains the ten experiments of the Computational Linguistics and Natural Language Processing Lab (CSEG4040P) performed by Aditya Khatri (SAP ID R2142230710, Roll No. 500119453), AI & ML, School of Computer Science, UPES Dehradun. Every experiment was written in Python, executed on a laptop and verified; the figures, numbers and console outputs in this report are the real results of those runs.")
para("Source code: https://github.com/Aditya-Khatri-7/NLP-Lab-Experiments")
para("Every program prints the line aditya khatri 500119453 in its banner and output lines, and every figure caption carries the same line.")
para("Structure of each experiment: Aim, Objectives, Tools and Dataset, Theory, Procedure, Source Code, Output, Figures, Observations and Result, Conclusion, Viva Questions.")
doc.add_heading("Summary of the experiments", level=2)
rows = [("Exp", "Title", "Key result"),
        ("1", "Employee CSV analysis", "Data Science has the highest average salary (97,253.53); Support the lowest (45,003.67)"),
        ("2", "Text retrieval and tokenization", "100 tweets saved; 160 sentences, 2,024 tokens, vocabulary 646"),
        ("3", "Pre-processing and morphology", "Tokens reduced by 54.5 %; lemmatization vs stemming compared"),
        ("4", "Text mining, corpus, NER", "Top-10 words, word cloud, Brown corpus (1.16 M words), 15 entities"),
        ("5", "POS tagging and chunking", "Twitter tagging; NP/VP/PP chunks; HMM-Viterbi accuracy 92.15 %"),
        ("6", "N-gram model and smoothing", "Kneser-Ney perplexity 220.18 (best); MLE 249,670"),
        ("7", "Sentiment analysis", "Logistic Regression 86.4 % vs VADER 63.5 %"),
        ("8", "Summarization and meaning", "TextRank best ROUGE-1 (0.447); Lesk WSD; WordNet similarity"),
        ("9 & 10", "Mini-project", "Review intelligence system: search, sentiment (85.75 %), summary, QA, topics")]
t = doc.add_table(rows=len(rows), cols=3); t.style = "Table Grid"
for i, row in enumerate(rows):
    for j, txt in enumerate(row):
        c = t.cell(i, j); c.width = Cm([1.6, 5.2, 9.6][j]); c.text = ""; run = c.paragraphs[0].add_run(txt); run.font.size = Pt(10); run.bold = (i == 0)
        c.paragraphs[0].paragraph_format.space_after = Pt(2)
        if i == 0: shade(c, "DBE6F5")
page_break()

# ================= EXPERIMENTS =================
for k, e in enumerate(EXPS, 1):
    label = "Experiments 9 & 10" if e["key"] == "Exp9_10" else f"Experiment {k}"
    doc.add_heading(f"{label}: {e['title']}", level=1)
    folder = os.path.join(ROOT, e["folder"]); fp_ = "9-10" if e["key"] == "Exp9_10" else str(k)
    doc.add_heading("Aim", level=2); para(e["aim"])
    doc.add_heading("Objectives", level=2); [item(x, "•") for x in e["objectives"]]
    doc.add_heading("Tools and Dataset", level=2); para(e["tools"])
    doc.add_heading("Theory", level=2); [para(x) for x in e["theory"]]
    doc.add_heading("Procedure / Algorithm", level=2); [item(x, f"{i}.") for i, x in enumerate(e["steps"], 1)]
    page_break(); doc.add_heading(f"Source Code - {e['script']}", level=2)
    src = open(os.path.join(folder, e["script"]), encoding="utf-8").read().rstrip().split("\n")
    mono([f"{i:3d}| {l.expandtabs(4)}" for i, l in enumerate(src, 1)], size=7)
    page_break(); doc.add_heading("Output (console)", level=2)
    para("The first lines of the real console output of the program (saved in outputs/console_output.txt):")
    co = [l for l in open(os.path.join(folder, "outputs", "console_output.txt"), encoding="utf-8").read().split("\n") if l.strip()][:30]
    tb = doc.add_table(rows=1, cols=1); tb.style = "Table Grid"; c = tb.cell(0, 0); shade(c, "F3F5F8"); c.text = ""
    first = True
    for ln in co:
        p = c.paragraphs[0] if first else c.add_paragraph(); first = False
        pf = p.paragraph_format; pf.space_after = Pt(0); pf.line_spacing = 1.0
        r = p.add_run(ln); r.font.name = "Consolas"; r.font.size = Pt(7)
    doc.add_heading("Output Figures", level=2)
    for i, (fn, c1, note) in enumerate(e["figs"], 1):
        figure(os.path.join(folder, "outputs", fn), f"Fig {fp_}.{i}: {c1} - {NAME}", note)
    doc.add_heading("Observations and Result", level=2); [item(x, "•") for x in e["obs"]]
    doc.add_heading("Conclusion", level=2); para(e["conclusion"])
    doc.add_heading("Viva Questions and Answers", level=2)
    for i, (q, a) in enumerate(e["viva"], 1):
        para(f"Q{i}. {q}", bold=True, after=1, keep=True); p = item(f"Ans: {a}", "")
    page_break()

doc.add_heading("Overall Conclusion", level=1)
para("In this lab the complete life cycle of a natural language processing system was implemented: data collection and cleaning (Experiments 1-3), exploration and information extraction (Experiment 4), linguistic analysis with tagging and chunking (Experiment 5), statistical language modelling with smoothing (Experiment 6), and the applications sentiment analysis (Experiment 7), summarization (Experiment 8) and an integrated mini-project (Experiments 9 and 10).")
para("Main learnings: (1) pre-processing choices such as stemming versus lemmatization change the quality of later steps; (2) probabilistic models such as HMMs and N-grams are strong but need smoothing for unseen events - Kneser-Ney was the best smoothing method here; (3) supervised TF-IDF models clearly outperform simple lexicon methods for sentiment; (4) extractive summarizers can be compared with proxy metrics, with TextRank being the best on overlap; (5) real applications combine many NLP components.")
para("Limitations and future work: spaCy models could not be used on the development laptop, so NLTK was used for NER and POS tagging; the corpora are small compared with modern data sets. Future work includes word embeddings, LSTM/GRU and transformer (BERT) models, and abstractive summarization.")
doc.add_heading("References", level=2)
for x in ["D. Jurafsky and J. H. Martin, Speech and Language Processing, 2nd/3rd Edition, Pearson/Prentice Hall.",
          "C. D. Manning and H. Schutze, Foundations of Statistical Natural Language Processing, MIT Press.",
          "S. Bird, E. Klein and E. Loper, Natural Language Processing with Python (NLTK Book), O'Reilly.",
          "R. Mihalcea and P. Tarau, TextRank: Bringing Order into Texts, EMNLP 2004.",
          "R. Kneser and H. Ney, Improved backing-off for M-gram language modeling, ICASSP 1995.",
          "C. J. Hutto and E. Gilbert, VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text, ICWSM 2014.",
          "Scikit-learn, NLTK, pandas, matplotlib and wordcloud documentation.", "CSEG4040P Course syllabus, School of Computer Science, UPES."]:
    item(x, "•")

doc.core_properties.author = "Aditya Khatri"; doc.core_properties.title = "NLP Lab Report - Aditya Khatri 500119453"
doc.save(OUT); print("saved", OUT, os.path.getsize(OUT) // 1024, "KB")
