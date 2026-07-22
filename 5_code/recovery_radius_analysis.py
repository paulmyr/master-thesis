"""Analysis for the multistage recovery benchmark (`results_recovery.csv`).

Headline claim: does the gradient multistage method (`grad_multistage`) match or
beat Nelder-Mead? We test it pairwise (paired across `direction_idx`) against
each NM baseline on the relaxed accuracy error (lower = better), and report the
secondary Γ as well.

Produces:
  - figures/recovery_accuracy_success.pdf   P(accuracy success) vs δ, per scenario
  - figures/recovery_accuracy_err.pdf        accuracy-error distribution per method
  - figures/recovery_gamma.pdf               median Γ vs δ (secondary)
  - stdout: paired Wilcoxon signed-rank + McNemar tables, and LaTeX rows.

Run with `python recovery_radius_analysis.py`.
"""

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).parent
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

METHOD = "grad_multistage"                       # the method under test
BASELINES = ["nm_multistage", "nm_naive", "grad_naive"]
METHOD_ORDER = [METHOD] + BASELINES

# --- Plot style (matches the thesis figure defaults) ------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 10, "axes.titlesize": 10, "axes.titleweight": "regular",
    "axes.linewidth": 0.6, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COLORS = {"grad_multistage": "#0072B2", "nm_multistage": "#E69F00",
          "nm_naive": "#D55E00", "grad_naive": "#999999"}
LABELS = {"grad_multistage": r"$\nabla$ multistage", "nm_multistage": "NM multistage",
          "nm_naive": "NM naive", "grad_naive": r"$\nabla$ naive"}
MARKERS = {"grad_multistage": "o", "nm_multistage": "^",
           "nm_naive": "v", "grad_naive": "s"}


# --- Load -------------------------------------------------------------------

def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


rows = list(csv.DictReader(open(ROOT / "results_recovery.csv")))
SCENARIOS = sorted({r["scenario"] for r in rows})
DELTAS = sorted({_f(r["delta"]) for r in rows})
METHODS = [m for m in METHOD_ORDER if m in {r["method"] for r in rows}]

# index[(scenario, method, delta, dir_idx)] -> row
index = {(r["scenario"], r["method"], _f(r["delta"]), int(r["direction_idx"])): r
         for r in rows}


def paired(metric, scenario, method_b, delta=None):
    """Return (method, baseline) arrays paired on direction_idx (and δ).

    Drops a pair if either side is missing / NaN / errored."""
    a_vals, b_vals = [], []
    deltas = [delta] if delta is not None else DELTAS
    dirs = sorted({int(r["direction_idx"]) for r in rows if r["scenario"] == scenario})
    for dl in deltas:
        for d in dirs:
            ra = index.get((scenario, METHOD, dl, d))
            rb = index.get((scenario, method_b, dl, d))
            if ra is None or rb is None:
                continue
            va, vb = _f(ra[metric]), _f(rb[metric])
            if not (np.isfinite(va) and np.isfinite(vb)):
                continue
            a_vals.append(va)
            b_vals.append(vb)
    return np.array(a_vals), np.array(b_vals)


def wilcoxon_report(a, b):
    """Paired signed-rank: H1 = method error < baseline error. Lower is better."""
    if len(a) < 6 or np.allclose(a, b):
        return float("nan"), float("nan"), float("nan")
    try:
        _, p = wilcoxon(a, b, alternative="less")
    except ValueError:
        p = float("nan")
    win = float(np.mean(a < b) + 0.5 * np.mean(a == b))  # fraction method better
    med_diff = float(np.median(a - b))                   # negative = method better
    return p, win, med_diff


def mcnemar_success(scenario, method_b, delta=None):
    """Paired binary success (accuracy_success). Returns (n10, n01, p)."""
    a, b = paired("accuracy_success", scenario, method_b, delta)
    n10 = int(np.sum((a == 1) & (b == 0)))   # method wins
    n01 = int(np.sum((a == 0) & (b == 1)))   # baseline wins
    n = n10 + n01
    p = binomtest(min(n10, n01), n, 0.5).pvalue if n > 0 else float("nan")
    return n10, n01, p


# --- Statistical tables -----------------------------------------------------

print("=" * 78)
print(f"PAIRED TESTS: {METHOD} vs baselines  (accuracy_err, lower = better)")
print("=" * 78)
print(f"{'scenario':<13}{'baseline':<16}{'n':>4}{'med Δerr':>10}"
      f"{'win%':>7}{'p(Wilcox)':>11}{'succ m/b':>10}{'p(McN)':>9}")
for scenario in SCENARIOS + ["POOLED"]:
    for mb in [m for m in BASELINES if m in METHODS]:
        if scenario == "POOLED":
            a = np.concatenate([paired("accuracy_err", s, mb)[0] for s in SCENARIOS])
            b = np.concatenate([paired("accuracy_err", s, mb)[1] for s in SCENARIOS])
            sa = sum(mcnemar_success(s, mb)[0] for s in SCENARIOS)
            sb = sum(mcnemar_success(s, mb)[1] for s in SCENARIOS)
            n = sa + sb
            pmc = binomtest(min(sa, sb), n, 0.5).pvalue if n > 0 else float("nan")
        else:
            a, b = paired("accuracy_err", scenario, mb)
            sa, sb, pmc = mcnemar_success(scenario, mb)
        p, win, med = wilcoxon_report(a, b)
        print(f"{scenario:<13}{mb:<16}{len(a):>4}{med:>10.2f}{100*win:>6.0f}%"
              f"{p:>11.4f}{f'{sa}/{sb}':>10}{pmc:>9.4f}")

print("\n--- secondary: Γ (higher = better; H1 = method Γ > baseline Γ) ---")
print(f"{'scenario':<13}{'baseline':<16}{'med ΔΓ':>9}{'win%':>7}{'p(Wilcox)':>11}")
for scenario in SCENARIOS + ["POOLED"]:
    for mb in [m for m in BASELINES if m in METHODS]:
        if scenario == "POOLED":
            a = np.concatenate([paired("gamma", s, mb)[0] for s in SCENARIOS])
            b = np.concatenate([paired("gamma", s, mb)[1] for s in SCENARIOS])
        else:
            a, b = paired("gamma", scenario, mb)
        # higher Γ better -> test (-a) < (-b)
        p, _, _ = wilcoxon_report(-a, -b)
        win = float(np.mean(a > b) + 0.5 * np.mean(a == b)) if len(a) else float("nan")
        med = float(np.median(a - b)) if len(a) else float("nan")
        print(f"{scenario:<13}{mb:<16}{med:>9.3f}{100*win:>6.0f}%{p:>11.4f}")

print("\n=== LaTeX rows (grad_multistage vs each baseline; paste into results table) ===")
for scenario in SCENARIOS:
    for mb in [m for m in BASELINES if m in METHODS]:
        a, b = paired("accuracy_err", scenario, mb)
        p, win, med = wilcoxon_report(a, b)
        sname = scenario.replace("_", " ")
        print(rf"{sname} & \texttt{{{mb.replace('_', chr(92)+'_')}}} & "
              rf"${med:+.2f}$ & ${100*win:.0f}\%$ & ${p:.3f}$ \\")


# --- Figures ----------------------------------------------------------------

def success_rate(scenario, method, delta):
    vals = [_f(index[k]["accuracy_success"])
            for k in index if k[0] == scenario and k[1] == method and k[2] == delta]
    vals = [v for v in vals if v in (0.0, 1.0)]
    return (sum(vals) / len(vals)) if vals else float("nan")


def median_metric(scenario, method, delta, metric):
    vals = [_f(index[k][metric])
            for k in index if k[0] == scenario and k[1] == method and k[2] == delta]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


# Fig 1: accuracy-success rate vs δ (headline)
fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(6.3, 3.0), sharey=True,
                         constrained_layout=True, squeeze=False)
for ax, scenario in zip(axes[0], SCENARIOS):
    for m in METHODS:
        ys = [success_rate(scenario, m, d) for d in DELTAS]
        ax.plot(DELTAS, ys, color=COLORS[m], marker=MARKERS[m], markersize=4.5,
                linewidth=1.3, label=LABELS[m])
    ax.set_xscale("log")
    ax.set_xticks(DELTAS)
    ax.set_xticklabels([f"{d:g}" for d in DELTAS])
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlabel(r"perturbation scale $\delta$")
    ax.set_title(scenario.replace("_", " "))
    ax.set_ylim(-0.04, 1.04)
axes[0][0].set_ylabel("P(accuracy success)")
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=len(METHODS))
fig.savefig(FIGS / "recovery_accuracy_success.pdf")
plt.close(fig)
print(f"\nwrote {FIGS / 'recovery_accuracy_success.pdf'}")

# Fig 2: accuracy-error distribution per method (pooled over δ), per scenario
fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(6.3, 3.0), sharey=True,
                         constrained_layout=True, squeeze=False)
for ax, scenario in zip(axes[0], SCENARIOS):
    data = []
    for m in METHODS:
        v = [_f(index[k]["accuracy_err"])
             for k in index if k[0] == scenario and k[1] == m]
        data.append([x for x in v if np.isfinite(x)])
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
    for patch, m in zip(bp["boxes"], METHODS):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xticks(range(1, len(METHODS) + 1))
    ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=30, ha="right")
    ax.set_title(scenario.replace("_", " "))
    ax.axhline(1.0, color="0.4", ls=":", lw=0.7)  # acc_err≈1 ~ within tolerance
axes[0][0].set_ylabel("accuracy error (lower = better)")
fig.savefig(FIGS / "recovery_accuracy_err.pdf")
plt.close(fig)
print(f"wrote {FIGS / 'recovery_accuracy_err.pdf'}")

# Fig 3: median Γ vs δ (secondary)
fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(6.3, 3.0), sharey=True,
                         constrained_layout=True, squeeze=False)
for ax, scenario in zip(axes[0], SCENARIOS):
    for m in METHODS:
        ys = [median_metric(scenario, m, d, "gamma") for d in DELTAS]
        ax.plot(DELTAS, ys, color=COLORS[m], marker=MARKERS[m], markersize=4.5,
                linewidth=1.3, label=LABELS[m])
    ax.axhline(0.5, color="0.4", ls=":", lw=0.7)
    ax.set_xscale("log")
    ax.set_xticks(DELTAS)
    ax.set_xticklabels([f"{d:g}" for d in DELTAS])
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlabel(r"perturbation scale $\delta$")
    ax.set_title(scenario.replace("_", " "))
axes[0][0].set_ylabel(r"median $\Gamma$")
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=len(METHODS))
fig.savefig(FIGS / "recovery_gamma.pdf")
plt.close(fig)
print(f"wrote {FIGS / 'recovery_gamma.pdf'}")
