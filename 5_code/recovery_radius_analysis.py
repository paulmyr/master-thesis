"""Analysis pass for the recovery-radius benchmark.

Reads `results_recovery.csv` + `results_init_floor.csv` and produces:
  - figures/recovery_radius_curves.pdf (P(Γ>0.5) vs δ, panel-per-scenario)
  - figures/recovery_per_parameter.pdf (median |Δθ|/range at δ=0.05)
  - tab:results-summary numbers printed to stdout for paste into LaTeX

Figures are sized to A4 textwidth (~6.3") with no further scaling expected
when included at `\\textwidth` in the thesis.
"""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
THESIS_FIGS = ROOT.parent / "2_thesis" / "figures"

DELTAS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
SCENARIOS = ["tonic", "adaptation", "initial_bursting"]
METHODS = ["grad_mse", "grad_guarino", "grad_vanrossum",
           "nm_guarino", "nm_guarino_hard"]
TRAINABLE = ["C_m", "g_L", "E_L", "v_T", "delta_T", "v_reset"]

# --- Plot style: thesis-grade defaults --------------------------------------
# Match LaTeX body text (Computer Modern-like serif via mathtext "cm").
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "regular",
    "axes.linewidth": 0.6,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.columnspacing": 1.4,
    "legend.handletextpad": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "pdf.fonttype": 42,   # embed TrueType, not Type-3
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Okabe–Ito colorblind-safe palette + neutral grey for the failing method.
COLORS = {
    "grad_mse":         "#999999",  # grey   – baseline that fails
    "grad_guarino":     "#0072B2",  # blue
    "grad_vanrossum":   "#009E73",  # bluish green
    "nm_guarino":       "#E69F00",  # orange
    "nm_guarino_hard":  "#D55E00",  # vermilion
    "init_floor":       "#000000",  # black, dashed
}
LABELS = {
    "grad_mse":         r"$\nabla$ MSE",
    "grad_guarino":     r"$\nabla$ Guarino",
    "grad_vanrossum":   r"$\nabla$ Van Rossum",
    "nm_guarino":       "NM (soft Guarino)",
    "nm_guarino_hard":  "NM (hard Guarino)",
    "init_floor":       "no-opt floor",
}
# Distinct markers so overlapping curves remain distinguishable.
MARKERS = {
    "grad_mse":         "o",
    "grad_guarino":     "s",
    "grad_vanrossum":   "D",
    "nm_guarino":       "^",
    "nm_guarino_hard":  "v",
    "init_floor":       "x",
}

# --- Load --------------------------------------------------------------------

def _load(path):
    return list(csv.DictReader(open(path)))


bench = _load(ROOT / "results_recovery.csv")
floor = _load(ROOT / "results_init_floor.csv")


def group_gammas(rows, default_method=None):
    """Group: (scenario, method, delta) -> list of γ values, one per direction."""
    g = defaultdict(list)
    for r in rows:
        m = default_method or r["method"]
        key = (r["scenario"], m, float(r["delta"]))
        try:
            g[key].append(float(r["gamma"]))
        except (ValueError, TypeError):
            g[key].append(float("nan"))
    return g


bench_gammas = group_gammas(bench)
floor_gammas = group_gammas(floor, default_method="init_floor")
all_gammas = {**bench_gammas, **floor_gammas}


def success_rate(scenario, method, delta):
    gs = all_gammas[(scenario, method, delta)]
    return sum(1 for g in gs if g > 0.5) / len(gs)


def _plot_curve(ax, scenario, method):
    ys = [success_rate(scenario, method, d) for d in DELTAS]
    is_floor = method == "init_floor"
    ax.plot(
        DELTAS, ys,
        color=COLORS[method],
        marker=MARKERS[method],
        markersize=4.5,
        markerfacecolor=COLORS[method] if not is_floor else "white",
        markeredgewidth=0.8,
        linewidth=1.2 if not is_floor else 1.0,
        linestyle="--" if is_floor else "-",
        label=LABELS[method],
        zorder=4 if method.startswith("nm_") else 3,
    )


# --- Figure 1: recovery-radius curves ---------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.1), sharey=True,
                         constrained_layout=True)
for ax, scenario in zip(axes, SCENARIOS):
    for m in METHODS + ["init_floor"]:
        _plot_curve(ax, scenario, m)
    ax.axhline(0.5, color="0.4", linestyle=":", linewidth=0.7, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks(DELTAS)
    ax.set_xticklabels([f"{d:g}" for d in DELTAS])
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlabel(r"perturbation scale $\delta$")
    ax.set_title(scenario.replace("_", " "))
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
axes[0].set_ylabel(r"$P(\Gamma > 0.5)$")

# Shared legend below the panels, two rows × three columns.
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
out1 = THESIS_FIGS / "recovery_radius_curves.pdf"
fig.savefig(out1)
plt.close(fig)
print(f"wrote {out1}")

# --- Figure 2: per-parameter recovery error at δ=0.05 -----------------------

# Pull param_err_<n> from the benchmark CSV, conditioned on δ=0.05.
err = defaultdict(list)  # (scenario, method, param) -> list of normalised abs errors
for r in bench:
    if float(r["delta"]) != 0.05:
        continue
    for n in TRAINABLE:
        v = r.get(f"param_err_{n}")
        if v not in (None, "", "nan"):
            err[(r["scenario"], r["method"], n)].append(float(v))

# Nicely typeset parameter labels (mathtext): C_m, g_L, E_L, V_T, Δ_T, V_r.
PARAM_TEX = {
    "C_m":     r"$C_m$",
    "g_L":     r"$g_L$",
    "E_L":     r"$E_L$",
    "v_T":     r"$V_T$",
    "delta_T": r"$\Delta_T$",
    "v_reset": r"$V_r$",
}

fig, axes = plt.subplots(3, 1, figsize=(6.3, 5.6), sharex=True,
                         constrained_layout=True)
x = np.arange(len(TRAINABLE))
width = 0.16  # 5 methods × 0.16 = 0.80 of the unit step → small inter-group gap

for ax, scenario in zip(axes, SCENARIOS):
    for i, m in enumerate(METHODS):
        meds = [
            median(err[(scenario, m, n)]) if err[(scenario, m, n)] else 0.0
            for n in TRAINABLE
        ]
        ax.bar(
            x + (i - 2) * width, meds, width=width,
            color=COLORS[m], edgecolor="white", linewidth=0.4,
            label=LABELS[m] if ax is axes[0] else None,
        )
    ax.set_title(scenario.replace("_", " "), loc="left", pad=4)
    ax.set_ylabel(r"median $|\Delta\theta|/$range")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
axes[-1].set_xticks(x)
axes[-1].set_xticklabels([PARAM_TEX[n] for n in TRAINABLE])
axes[-1].set_xlabel("parameter")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
out2 = THESIS_FIGS / "recovery_per_parameter.pdf"
fig.savefig(out2)
plt.close(fig)
print(f"wrote {out2}")

# --- Table numbers (paste into 5_results.tex) -------------------------------

print("\n=== tab:results-summary numbers ===")
print(f"{'method':<18} {'tonic d50/Γ_05':<22} {'adapt d50/Γ_05':<22} {'init_b d50/Γ_05':<22}")


def d50(scenario, method):
    """Largest grid value of δ with success rate ≥ 0.5; '--' if none."""
    last = None
    for d in DELTAS:
        if success_rate(scenario, method, d) >= 0.5:
            last = d
    return f"{last:g}" if last else "--"


def gmed(scenario, method, d=0.05):
    return median(all_gammas[(scenario, method, d)])


for m in METHODS + ["init_floor"]:
    cells = []
    for s in SCENARIOS:
        cells.append(f"{d50(s, m):>5} / {gmed(s, m):+.2f}")
    print(f"{m:<18}  " + "   ".join(c.ljust(20) for c in cells))

print("\n=== LaTeX rows (paste into tab:results-summary) ===")
for m in METHODS + ["init_floor"]:
    cells = []
    for s in SCENARIOS:
        cells.append(f"${d50(s, m)}$ & ${gmed(s, m):+.2f}$")
    name = m.replace("_", r"\_")
    print(rf"\texttt{{{name}}}      & " + " & ".join(cells) + r" \\")
