#!/usr/bin/env python3
"""
Generate a comprehensive study guide PDF on backpropagation,
with focus on surrogate gradients for spiking neuron models.
"""

import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np
from textwrap import dedent

# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "mathtext.fontset": "cm",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "text.usetex": False,
})

A4_W, A4_H = 8.27, 11.69  # inches
MARGIN = 0.08  # fraction

COLOR_PRIMARY = "#1a1a2e"
COLOR_ACCENT = "#e94560"
COLOR_BLUE = "#0f3460"
COLOR_TEAL = "#16213e"
COLOR_LIGHT = "#f5f5f5"
COLOR_GREEN = "#2d6a4f"
COLOR_ORANGE = "#e76f51"


"""Figure: Gradient magnitude vs timestep."""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(A4_W, 3.2))

# Left: gradient decay factor
T = 5000
dt = 0.025  # ms
g_L_values = [10, 12, 15]  # nS
C_m = 200  # pF

t = np.arange(T)
for g_L in g_L_values:
    factor = 1 - dt * g_L / C_m
    decay = factor ** t
    ax1.semilogy(t * dt, decay, label=f"$g_L$ = {g_L} nS", lw=1.5)

ax1.set_xlabel("Time backward from loss (ms)")
ax1.set_ylabel("Gradient magnitude (relative)")
ax1.set_title("(a) Gradient decay through leak", fontsize=10)
ax1.legend(fontsize=8)
ax1.set_xlim(0, T*dt)
ax1.grid(True, alpha=0.3)

# Right: effective gradient contribution by timestep position
# For timestep at position t, gradient must travel (T-1-t) steps backward
# to reach the loss, so it decays by factor^(T-1-t)
g_L = 10
factor = 1 - dt * g_L / C_m
steps_backward = T - 1 - np.arange(T)
decay = factor ** steps_backward
# Contribution = how much a timestep's local gradient contributes to total
contribution = decay / decay.sum()

ax2.fill_between(np.arange(T) * dt, contribution * 1000, alpha=0.3, color=COLOR_BLUE)
ax2.plot(np.arange(T) * dt, contribution * 1000, color=COLOR_BLUE, lw=1)
ax2.set_xlabel("Timestep position (ms)")
ax2.set_ylabel("Relative contribution (\u2030)")
ax2.set_title("(b) Per-timestep gradient contribution", fontsize=10)
ax2.set_xlim(0, T*dt)
ax2.grid(True, alpha=0.3)
ax2.annotate("Late timesteps\ndominate gradient",
            xy=(T*dt*0.9, contribution[-100]*1000),
            xytext=(T*dt*0.5, contribution.max()*1000*0.8),
            arrowprops=dict(arrowstyle="->", color=COLOR_ACCENT),
            fontsize=8, color=COLOR_ACCENT)

fig.tight_layout()
fig.savefig("vanishing-gradients.pdf")
