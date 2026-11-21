"""Shared helpers for NLP Lab experiments. Author: aditya khatri 500119453"""
import os, sys, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAME = "aditya khatri 500119453"
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
    "font.size": 10, "axes.titleweight": "bold", "axes.titlesize": 13})
PAL = ["#3B6FB6", "#E07A3F", "#4FA37A", "#C45A7C", "#8B6FC0", "#C9A227", "#4AA3B8", "#7A7A7A"]

def banner(title):
    line = "=" * 70
    print(f"{line}\n{title}\n{NAME}\n{line}")

def say(*parts):
    """print a line tagged with the author name"""
    print(f"[{NAME}]", *parts)

def section(title):
    print(f"\n--- {title} --- [{NAME}]")

def out_dir(file):
    d = os.path.join(os.path.dirname(os.path.abspath(file)), "outputs")
    os.makedirs(d, exist_ok=True); return d

def save(fig, folder, fname, caption):
    """Save figure with a caption line containing the author name."""
    fig.text(0.5, 0.005, f"{caption}  |  {NAME}", ha="center", va="bottom",
             fontsize=9, style="italic", color="#444")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = os.path.join(folder, fname); fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig); say("saved figure ->", fname)
    return p
