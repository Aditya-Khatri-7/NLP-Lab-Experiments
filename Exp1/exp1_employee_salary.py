"""EXPERIMENT 1: Load employee CSV and plot average salary department-wise.
Author: aditya khatri 500119453
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common import *

banner("EXPERIMENT 1 - Employee CSV analysis (average salary by department)")
here = os.path.dirname(os.path.abspath(__file__)); OUT = out_dir(__file__)

# ---- 1. create a sample employees.csv (only if absent) ----
csv_path = os.path.join(here, "employees.csv")
if not os.path.exists(csv_path):
    rng = np.random.default_rng(500119453)
    depts = {"Engineering": 85000, "Data Science": 95000, "HR": 52000,
             "Marketing": 60000, "Finance": 72000, "Sales": 58000, "Support": 45000}
    rows = []
    for i in range(1, 121):
        d = rng.choice(list(depts))
        rows.append([1000 + i, f"Employee_{i:03d}", d, int(rng.integers(1, 20)),
                     int(rng.normal(depts[d], depts[d] * .12))])
    pd.DataFrame(rows, columns=["EmpID", "Name", "Department", "Experience", "Salary"]).to_csv(csv_path, index=False)
    say("sample employees.csv created")

# ---- 2. load + explore ----
df = pd.read_csv(csv_path)
section("First 5 rows"); print(df.head().to_string(index=False))
section("Shape / missing values")
say("shape:", df.shape); print(df.isna().sum().to_string())
section("Descriptive statistics"); print(df["Salary"].describe().round(2).to_string())

# ---- 3. average salary per department ----
avg = df.groupby("Department")["Salary"].mean().sort_values(ascending=False)
cnt = df.groupby("Department")["Salary"].count()
section("Average salary by department")
for d, v in avg.items(): say(f"{d:14s} avg = {v:10,.2f}   employees = {cnt[d]}")
avg.round(2).to_csv(os.path.join(OUT, "avg_salary_by_department.csv"))

# ---- 4. graphs ----
fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.bar(avg.index, avg.values, color=PAL[:len(avg)], edgecolor="white")
for b, v in zip(bars, avg.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 800, f"{v:,.0f}", ha="center", fontsize=9)
ax.axhline(df.Salary.mean(), ls="--", c="#444", lw=1, label=f"Company mean = {df.Salary.mean():,.0f}")
ax.set(title="Average Salary by Department", xlabel="Department", ylabel="Average salary")
ax.legend(); plt.setp(ax.get_xticklabels(), rotation=25)
save(fig, OUT, "avg_salary_bar.png", "Fig 1: Department-wise average salary")

fig, ax = plt.subplots(figsize=(10, 5.5))
order = list(avg.index)
ax.boxplot([df[df.Department == d].Salary for d in order], tick_labels=order, patch_artist=True,
           boxprops=dict(facecolor="#9db8e0"), medianprops=dict(color="#c0392b"))
ax.set(title="Salary Distribution by Department", ylabel="Salary"); plt.setp(ax.get_xticklabels(), rotation=25)
save(fig, OUT, "salary_boxplot.png", "Fig 2: Salary spread inside each department")

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.scatter(df.Experience, df.Salary, c=[PAL[order.index(d) % 8] for d in df.Department], alpha=.7)
m, c = np.polyfit(df.Experience, df.Salary, 1); xs = np.array([df.Experience.min(), df.Experience.max()])
ax.plot(xs, m * xs + c, "k--", label=f"trend: +{m:,.0f} per year")
ax.set(title="Experience vs Salary", xlabel="Years of experience", ylabel="Salary"); ax.legend()
save(fig, OUT, "experience_vs_salary.png", "Fig 3: Experience vs salary with fitted trend line")

say("Highest paying department:", avg.index[0], "| Lowest:", avg.index[-1])
say("EXPERIMENT 1 COMPLETED")
