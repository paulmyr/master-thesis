#!/usr/bin/env python3
"""
Generate a comprehensive study guide PDF on backpropagation,
with focus on surrogate gradients for spiking neuron models.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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


class PageWriter:
    """Multi-page text writer that auto-paginates when content overflows."""

    BOTTOM_MARGIN = 0.045

    def __init__(self, title=None, subtitle=None):
        self.figures = []
        self.title = title
        self.subtitle = subtitle
        self._new_page(title, subtitle)

    def _new_page(self, title=None, subtitle=None):
        fig = plt.figure(figsize=(A4_W, A4_H))
        ax = fig.add_axes([MARGIN, MARGIN, 1 - 2*MARGIN, 1 - 2*MARGIN])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.y = 0.97
        self.figures.append(fig)
        if title:
            self.ax.text(0.0, self.y, title, fontsize=16, fontweight="bold",
                         color=COLOR_PRIMARY, va="top")
            self.y -= 0.04
            self.ax.axhline(y=self.y, xmin=0, xmax=1, color=COLOR_ACCENT,
                            linewidth=2)
            self.y -= 0.02
        if subtitle:
            self.ax.text(0.0, self.y, subtitle, fontsize=12, fontstyle="italic",
                         color=COLOR_BLUE, va="top")
            self.y -= 0.03

    def _ensure_space(self, needed=0.04):
        """Start a new continuation page if not enough vertical space."""
        if self.y < self.BOTTOM_MARGIN + needed:
            cont_title = self.title + " (cont.)" if self.title else None
            self._new_page(cont_title)

    def text(self, text, fontsize=10, color=COLOR_PRIMARY, indent=0.0,
             style="normal", weight="normal", line_spacing=0.022):
        import textwrap
        lines = text.split("\n")
        for line in lines:
            if not line.strip():
                self.y -= line_spacing * 0.6
                continue
            wrap_width = int(90 * (1 - indent))
            wrapped = (textwrap.wrap(line, width=wrap_width)
                       if len(line) > wrap_width else [line])
            for wl in wrapped:
                self._ensure_space(line_spacing + 0.005)
                self.ax.text(indent, self.y, wl, fontsize=fontsize, color=color,
                             va="top", fontstyle=style, fontweight=weight,
                             transform=self.ax.transAxes)
                self.y -= line_spacing

    def equation(self, eq_text, fontsize=13):
        self._ensure_space(0.055)
        self.y -= 0.01
        self.ax.text(0.5, self.y, eq_text, fontsize=fontsize,
                     color=COLOR_PRIMARY, va="top", ha="center",
                     transform=self.ax.transAxes)
        self.y -= 0.035

    def bullet(self, text, fontsize=10, indent=0.03):
        self.text("  \u2022  " + text, fontsize=fontsize, indent=indent)

    def gap(self, size=0.005):
        self.y -= size


def new_text_page(title=None, subtitle=None):
    """Create a blank A4 page for text content (legacy single-page API)."""
    fig = plt.figure(figsize=(A4_W, A4_H))
    ax = fig.add_axes([MARGIN, MARGIN, 1 - 2*MARGIN, 1 - 2*MARGIN])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y = 0.97
    if title:
        ax.text(0.0, y, title, fontsize=16, fontweight="bold",
                color=COLOR_PRIMARY, va="top")
        y -= 0.04
        ax.axhline(y=y, xmin=0, xmax=1, color=COLOR_ACCENT, linewidth=2)
        y -= 0.02
    if subtitle:
        ax.text(0.0, y, subtitle, fontsize=12, fontstyle="italic",
                color=COLOR_BLUE, va="top")
        y -= 0.03
    return fig, ax, y


def add_text(ax, y, text, fontsize=10, color=COLOR_PRIMARY, indent=0.0,
             style="normal", weight="normal", line_spacing=0.022):
    """Add wrapped text lines to the page. Returns new y position."""
    import textwrap
    lines = text.split("\n")
    for line in lines:
        if not line.strip():
            y -= line_spacing * 0.6
            continue
        wrap_width = int(90 * (1 - indent))
        wrapped = textwrap.wrap(line, width=wrap_width) if len(line) > wrap_width else [line]
        for wl in wrapped:
            if y < 0.03:
                return y
            ax.text(indent, y, wl, fontsize=fontsize, color=color,
                    va="top", fontstyle=style, fontweight=weight,
                    transform=ax.transAxes)
            y -= line_spacing
    return y


def add_equation(ax, y, eq_text, fontsize=13):
    """Add a centered LaTeX equation."""
    y -= 0.01
    ax.text(0.5, y, eq_text, fontsize=fontsize, color=COLOR_PRIMARY,
            va="top", ha="center", transform=ax.transAxes)
    y -= 0.035
    return y


def add_bullet(ax, y, text, fontsize=10, bullet="  \u2022  ", indent=0.03):
    """Add a bullet point."""
    full = bullet + text
    return add_text(ax, y, full, fontsize=fontsize, indent=indent)


# ===========================================================================
# PAGE GENERATORS
# ===========================================================================

def make_title_page():
    fig = plt.figure(figsize=(A4_W, A4_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background accent
    rect = FancyBboxPatch((0.05, 0.55), 0.9, 0.35, boxstyle="round,pad=0.02",
                          facecolor=COLOR_PRIMARY, edgecolor="none", alpha=0.95)
    ax.add_patch(rect)

    ax.text(0.5, 0.82, "Backpropagation", fontsize=32, fontweight="bold",
            color="white", ha="center", va="center")
    ax.text(0.5, 0.75, "From Fundamentals to Surrogate Gradients",
            fontsize=18, color=COLOR_ACCENT, ha="center", va="center")
    ax.text(0.5, 0.68, "A Study Guide for Gradient-Based Optimization\nof Spiking Neuron Models",
            fontsize=13, color="#cccccc", ha="center", va="center", linespacing=1.5)
    ax.text(0.5, 0.60, "Paul Mayer  \u2014  KTH Royal Institute of Technology",
            fontsize=11, color="#aaaaaa", ha="center", va="center")

    # Table of contents
    ax.text(0.5, 0.45, "Contents", fontsize=16, fontweight="bold",
            color=COLOR_PRIMARY, ha="center", va="center")
    toc = [
        ("1", "Mathematical Prerequisites: Derivatives and the Chain Rule"),
        ("2", "Computational Graphs and Automatic Differentiation"),
        ("3", "The Backpropagation Algorithm"),
        ("4", "Backpropagation Through Time (BPTT)"),
        ("4b", "Why the Loss Function Must Be Differentiable"),
        ("4c", "BPTT in Detail: From Loss to Parameter Update"),
        ("4d", "Spike-Gated Gradients"),
        ("4e", "From Loss to Parameters: The Complete Chain"),
        ("5", "The Spike Discontinuity Problem"),
        ("6", "Surrogate Gradient Methods"),
        ("7", "Loss Function Design for Spiking Models"),
        ("8", "Putting It All Together: The Full Gradient Pipeline"),
        ("", "References"),
    ]
    y = 0.40
    for num, title in toc:
        label = f"{num}.  {title}" if num else f"     {title}"
        ax.text(0.15, y, label, fontsize=9, color=COLOR_BLUE, va="center")
        y -= 0.025

    return fig


def make_ch1_page1():
    """Chapter 1: Mathematical Prerequisites - Derivatives."""
    fig, ax, y = new_text_page("1. Mathematical Prerequisites",
                                "Derivatives and the Chain Rule")

    y = add_text(ax, y, dedent("""\
    Backpropagation is, at its core, a systematic application of the chain rule from \
    multivariable calculus. Before diving into the algorithm, we review the essential \
    mathematical machinery."""), indent=0.0)
    y -= 0.01

    y = add_text(ax, y, "1.1  Scalar Derivatives", fontsize=12, weight="bold")
    y = add_text(ax, y, dedent("""\
    For a function f: R \u2192 R, the derivative f'(x) measures the instantaneous rate of \
    change. Geometrically, it is the slope of the tangent line at x. If f(x) = x\u00b2, then \
    f'(x) = 2x. This tells us: a small perturbation \u03b5 at x changes f by approximately \
    2x\u00b7\u03b5 (Rudin, 1976, Ch. 5)."""))
    y -= 0.005

    y = add_text(ax, y, "1.2  Partial Derivatives and Gradients", fontsize=12, weight="bold")
    y = add_text(ax, y, dedent("""\
    For f: R\u207f \u2192 R, the partial derivative \u2202f/\u2202x\u1d62 measures the rate of change along \
    the i-th coordinate while holding all others fixed. The gradient collects all partial \
    derivatives into a vector:"""))
    y = add_equation(ax, y, r"$\nabla f(\mathbf{x}) = \left(\frac{\partial f}{\partial x_1},\; \frac{\partial f}{\partial x_2},\; \ldots,\; \frac{\partial f}{\partial x_n}\right)$")

    y = add_text(ax, y, dedent("""\
    The gradient points in the direction of steepest ascent. To minimize f, we move in \
    the opposite direction: x \u2190 x \u2212 \u03b7\u2207f(x), where \u03b7 is the learning rate. This is \
    gradient descent (Nocedal & Wright, 2006, Ch. 2)."""))
    y -= 0.005

    y = add_text(ax, y, "1.3  The Chain Rule", fontsize=12, weight="bold")
    y = add_text(ax, y, dedent("""\
    The chain rule is the fundamental theorem enabling backpropagation. For composed \
    functions f(g(x)), the derivative of the composition is the product of derivatives:"""))
    y = add_equation(ax, y, r"$\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$")

    y = add_text(ax, y, dedent("""\
    In higher dimensions, this generalizes to the Jacobian product. If f: R\u1d50 \u2192 R\u1d56 and \
    g: R\u207f \u2192 R\u1d50, then (Goodfellow et al., 2016, Ch. 6.5):"""))
    y = add_equation(ax, y, r"$\frac{\partial (f \circ g)}{\partial \mathbf{x}} = \frac{\partial f}{\partial \mathbf{g}} \cdot \frac{\partial \mathbf{g}}{\partial \mathbf{x}}$", fontsize=12)

    y = add_text(ax, y, dedent("""\
    where \u2202f/\u2202g is the p\u00d7m Jacobian of f and \u2202g/\u2202x is the m\u00d7n Jacobian of g. The \
    result is a p\u00d7n Jacobian. For scalar-valued loss functions (p=1), this simplifies to \
    vector-Jacobian products (VJPs), which are what JAX computes."""))
    y -= 0.005

    y = add_text(ax, y, "1.4  Why the Chain Rule Matters for Optimization", fontsize=12, weight="bold")
    y = add_text(ax, y, dedent("""\
    In machine learning, we have a loss function L(\u03b8) that depends on parameters \u03b8 through \
    a long sequence of composed operations. The chain rule lets us decompose the gradient \
    into local derivatives at each operation, computed and multiplied together. This is \
    computationally tractable even for millions of parameters \u2014 we never need to perturb \
    each parameter individually (which would require O(n) forward passes)."""))

    return fig


def make_ch1_chain_rule_figure():
    """Visual diagram of the chain rule."""
    fig, axes = plt.subplots(1, 2, figsize=(A4_W, 4.5))
    fig.suptitle("Figure 1.1: The Chain Rule Visualized", fontsize=12, fontweight="bold", y=0.98)

    # Left: simple chain
    ax = axes[0]
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("(a) Simple chain: f(g(x))", fontsize=10)
    ax.axis("off")

    nodes = [(0, 0, "$x$"), (1.5, 0, "$g(x)$"), (3, 0, "$f(g(x))$")]
    for x, yy, label in nodes:
        circle = plt.Circle((x, yy), 0.35, facecolor=COLOR_LIGHT,
                            edgecolor=COLOR_PRIMARY, linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, yy, label, ha="center", va="center", fontsize=11)

    # Forward arrows (top)
    ax.annotate("", xy=(1.1, 0.15), xytext=(0.4, 0.15),
                arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=1.5))
    ax.text(0.75, 0.35, "$g$", fontsize=10, ha="center", color=COLOR_BLUE)

    ax.annotate("", xy=(2.55, 0.15), xytext=(1.95, 0.15),
                arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=1.5))
    ax.text(2.25, 0.35, "$f$", fontsize=10, ha="center", color=COLOR_BLUE)

    # Backward arrows (bottom)
    ax.annotate("", xy=(0.4, -0.15), xytext=(1.1, -0.15),
                arrowprops=dict(arrowstyle="->", color=COLOR_ACCENT, lw=1.5))
    ax.text(0.75, -0.5, "$g'(x)$", fontsize=10, ha="center", color=COLOR_ACCENT)

    ax.annotate("", xy=(1.95, -0.15), xytext=(2.55, -0.15),
                arrowprops=dict(arrowstyle="->", color=COLOR_ACCENT, lw=1.5))
    ax.text(2.25, -0.5, "$f'(g(x))$", fontsize=10, ha="center", color=COLOR_ACCENT)

    ax.text(1.5, -1.1, r"$\frac{df}{dx} = f'(g(x)) \cdot g'(x)$",
            fontsize=12, ha="center", color=COLOR_PRIMARY,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3e0", edgecolor=COLOR_ORANGE))

    # Right: multivariate chain
    ax2 = axes[1]
    ax2.set_xlim(-0.5, 4)
    ax2.set_ylim(-2, 2)
    ax2.set_title("(b) Multivariate: paths sum", fontsize=10)
    ax2.axis("off")

    # Input
    c = plt.Circle((0, 0), 0.35, facecolor=COLOR_LIGHT,
                   edgecolor=COLOR_PRIMARY, linewidth=1.5)
    ax2.add_patch(c)
    ax2.text(0, 0, "$x$", ha="center", va="center", fontsize=12)

    # Two intermediate
    for yy, label in [(1, "$a$"), (-1, "$b$")]:
        c = plt.Circle((1.8, yy), 0.35, facecolor=COLOR_LIGHT,
                       edgecolor=COLOR_PRIMARY, linewidth=1.5)
        ax2.add_patch(c)
        ax2.text(1.8, yy, label, ha="center", va="center", fontsize=12)

    # Output
    c = plt.Circle((3.5, 0), 0.35, facecolor=COLOR_LIGHT,
                   edgecolor=COLOR_PRIMARY, linewidth=1.5)
    ax2.add_patch(c)
    ax2.text(3.5, 0, "$L$", ha="center", va="center", fontsize=12)

    # Arrows
    for yy in [1, -1]:
        ax2.annotate("", xy=(1.4, yy*0.7), xytext=(0.35, yy*0.15),
                     arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=1.2))
        ax2.annotate("", xy=(3.1, yy*0.15), xytext=(2.2, yy*0.7),
                     arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=1.2))

    ax2.text(1.75, -1.7,
             r"$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial a}\frac{\partial a}{\partial x} + \frac{\partial L}{\partial b}\frac{\partial b}{\partial x}$",
             fontsize=11, ha="center", color=COLOR_PRIMARY,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", edgecolor=COLOR_GREEN))

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def make_ch2_page1():
    """Chapter 2: Computational Graphs."""
    pw = PageWriter("2. Computational Graphs and Automatic Differentiation")

    pw.text(dedent("""\
    Modern deep learning frameworks (JAX, PyTorch, TensorFlow) do not require users to \
    derive gradients by hand. Instead, they use automatic differentiation (AD), which \
    mechanically applies the chain rule through a computational graph \
    (Baydin et al., 2018)."""))
    pw.gap()

    pw.text("2.1  What is a Computational Graph?", fontsize=12, weight="bold")
    pw.text(dedent("""\
    A computational graph is a directed acyclic graph (DAG) where each node represents \
    an elementary operation (addition, multiplication, exp, etc.) and edges represent data \
    flow. Any computable function can be decomposed into such a graph."""))
    pw.gap()

    pw.text("Example: f(x, y) = (x + y) \u00b7 sin(x) decomposes into:")
    pw.bullet("a = x + y       (addition)")
    pw.bullet("b = sin(x)      (sine)")
    pw.bullet("f = a \u00b7 b        (multiplication)")
    pw.gap()

    pw.text("Each operation has a known local derivative. The chain rule chains them together.")
    pw.gap()

    pw.text("2.2  Forward Mode vs. Reverse Mode AD", fontsize=12, weight="bold")
    pw.text("There are two ways to apply the chain rule through a computational graph:")
    pw.gap()

    pw.text("Forward mode (tangent propagation):", weight="bold", indent=0.02)
    pw.text(dedent("""\
    Propagates derivatives from inputs to outputs. Computes one column of the Jacobian \
    per pass. Efficient when there are few inputs and many outputs. Used via jax.jvp()."""),
            indent=0.04)
    pw.gap()

    pw.text("Reverse mode (adjoint propagation):", weight="bold", indent=0.02)
    pw.text(dedent("""\
    Propagates derivatives from outputs to inputs. Computes one row of the Jacobian per \
    pass. Efficient when there are many inputs and few outputs \u2014 exactly the situation in \
    machine learning, where we have millions of parameters but a single scalar loss. \
    Backpropagation IS reverse-mode AD (Griewank & Walther, 2008)."""), indent=0.04)
    pw.gap()

    pw.text("2.3  JAX's Approach: Tracing and Transformation", fontsize=12, weight="bold")
    pw.text("JAX implements reverse-mode AD via program tracing. When you call jax.grad(f)(x), JAX:")
    pw.bullet("Traces f(x) by executing it with abstract values, recording every operation")
    pw.bullet("Builds an internal representation (jaxpr) of the computation graph")
    pw.bullet("Transforms the jaxpr into a backward-pass program using VJP rules")
    pw.bullet("Executes both forward and backward passes to return the gradient")
    pw.gap()

    pw.text(dedent("""\
    Each JAX primitive (jnp.add, jnp.exp, jnp.matmul, ...) has a registered VJP rule. \
    The custom_vjp decorator (used in surrogate gradients) lets users override these rules \
    for specific functions (Bradbury et al., 2018)."""))
    pw.gap()

    pw.text("2.4  Vector-Jacobian Products (VJPs)", fontsize=12, weight="bold")
    pw.text(dedent("""\
    In reverse mode, we never build the full Jacobian matrix. Instead, we compute \
    vector-Jacobian products. Given upstream gradient v (a row vector) and local Jacobian J:"""))
    pw.equation(r"$\mathbf{v}^\top \mathbf{J} = \mathbf{v}^\top \frac{\partial \mathbf{f}}{\partial \mathbf{x}}$")
    pw.text(dedent("""\
    This costs O(m\u00b7n) \u2014 the same as one matrix-vector multiply \u2014 regardless of the \
    number of outputs. For a scalar loss, the initial upstream gradient is simply 1.0."""))

    return pw.figures


def make_ch3_page1():
    """Chapter 3: The Backpropagation Algorithm."""
    pw = PageWriter("3. The Backpropagation Algorithm")

    pw.text(dedent("""\
    Backpropagation (Rumelhart, Hinton & Williams, 1986) is the standard algorithm for \
    computing gradients in neural networks. It is simply reverse-mode automatic \
    differentiation applied to a specific computational graph."""))
    pw.gap()

    pw.text("3.1  Setup: A Simple Neural Network", fontsize=12, weight="bold")
    pw.text("Consider a two-layer network with weights W\u2081, W\u2082 and activation \u03c3:")
    pw.equation(r"$\mathbf{h} = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \qquad \hat{\mathbf{y}} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$")
    pw.equation(r"$L = \frac{1}{2}\|\hat{\mathbf{y}} - \mathbf{y}\|^2$")

    pw.text("3.2  Forward Pass", fontsize=12, weight="bold")
    pw.text("Evaluate each operation in order, storing intermediate values (activations):")
    pw.bullet("z\u2081 = W\u2081x + b\u2081                    (linear)")
    pw.bullet("h  = \u03c3(z\u2081)                        (activation)")
    pw.bullet("z\u2082 = W\u2082h + b\u2082                    (linear)")
    pw.bullet("L  = \u00bd\u2016z\u2082 \u2212 y\u2016\u00b2                   (loss)")
    pw.gap()

    pw.text("3.3  Backward Pass", fontsize=12, weight="bold")
    pw.text("Starting from the loss, propagate gradients backward using local derivatives:")
    pw.gap()

    pw.text("Step 1: Loss gradient", indent=0.02, weight="bold")
    pw.equation(r"$\frac{\partial L}{\partial \mathbf{z}_2} = \hat{\mathbf{y}} - \mathbf{y}$")

    pw.text("Step 2: Through second linear layer", indent=0.02, weight="bold")
    pw.equation(r"$\frac{\partial L}{\partial \mathbf{W}_2} = \frac{\partial L}{\partial \mathbf{z}_2} \mathbf{h}^\top, \qquad \frac{\partial L}{\partial \mathbf{h}} = \mathbf{W}_2^\top \frac{\partial L}{\partial \mathbf{z}_2}$")

    pw.text("Step 3: Through activation", indent=0.02, weight="bold")
    pw.equation(r"$\frac{\partial L}{\partial \mathbf{z}_1} = \frac{\partial L}{\partial \mathbf{h}} \odot \sigma'(\mathbf{z}_1)$")

    pw.text("Step 4: Through first linear layer", indent=0.02, weight="bold")
    pw.equation(r"$\frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial \mathbf{z}_1} \mathbf{x}^\top$")

    pw.gap()
    pw.text(dedent("""\
    Key insight: each step only requires (a) the upstream gradient, (b) the locally stored \
    activation, and (c) the known derivative of the local operation. The computational cost \
    is roughly twice that of a single forward pass (Goodfellow et al., 2016, Ch. 6.5)."""))

    pw.gap()
    pw.text("3.4  Gradient Descent Update", fontsize=12, weight="bold")
    pw.text("Once we have all gradients, parameters are updated (for vanilla SGD):")
    pw.equation(r"$\theta \leftarrow \theta - \eta \, \nabla_\theta L$")
    pw.text(dedent("""\
    Modern optimizers like Adam (Kingma & Ba, 2015) maintain per-parameter running averages \
    of gradients (momentum) and squared gradients (RMS scaling), adapting the effective \
    learning rate for each parameter."""))

    return pw.figures


def make_ch3_backprop_figure():
    """Figure: Forward and backward pass through a network."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(A4_W, 5.5))
    fig.suptitle("Figure 3.1: Forward and Backward Pass Through a Two-Layer Network",
                 fontsize=11, fontweight="bold", y=0.99)

    for ax, title, color, direction in [
        (ax1, "Forward Pass", COLOR_BLUE, "right"),
        (ax2, "Backward Pass", COLOR_ACCENT, "left")
    ]:
        ax.set_xlim(-0.5, 6)
        ax.set_ylim(-0.8, 1.2)
        ax.set_title(title, fontsize=10, color=color, fontweight="bold")
        ax.axis("off")

        boxes = [
            (0, "$\\mathbf{x}$", "Input"),
            (1.2, "$\\mathbf{z}_1$", "Linear"),
            (2.4, "$\\mathbf{h}$", "$\\sigma$"),
            (3.6, "$\\mathbf{z}_2$", "Linear"),
            (4.8, "$L$", "Loss"),
        ]

        for x, label, sublabel in boxes:
            rect = FancyBboxPatch((x-0.35, -0.3), 0.7, 0.6,
                                  boxstyle="round,pad=0.05",
                                  facecolor=COLOR_LIGHT, edgecolor=color, linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x, 0.1, label, ha="center", va="center", fontsize=12)
            ax.text(x, -0.5, sublabel, ha="center", va="center", fontsize=8, color="gray")

        # Arrows
        for i in range(len(boxes)-1):
            x1 = boxes[i][0] + 0.4
            x2 = boxes[i+1][0] - 0.4
            if direction == "right":
                ax.annotate("", xy=(x2, 0.1), xytext=(x1, 0.1),
                           arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
            else:
                ax.annotate("", xy=(x1, 0.1), xytext=(x2, 0.1),
                           arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

        # Labels on arrows
        if direction == "right":
            labels = ["$\\mathbf{W}_1, \\mathbf{b}_1$", "$\\sigma(\\cdot)$",
                      "$\\mathbf{W}_2, \\mathbf{b}_2$", "$\\|\\cdot\\|^2$"]
        else:
            labels = [
                "$\\frac{\\partial L}{\\partial \\mathbf{z}_1}$",
                "$\\odot\\,\\sigma'$",
                "$\\frac{\\partial L}{\\partial \\mathbf{h}}$",
                "$\\hat{\\mathbf{y}}-\\mathbf{y}$"
            ]
        for i, lbl in enumerate(labels):
            xm = (boxes[i][0] + boxes[i+1][0]) / 2
            ax.text(xm, 0.55, lbl, ha="center", va="center", fontsize=9, color=color)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def make_ch4_page1():
    """Chapter 4: BPTT."""
    pw = PageWriter("4. Backpropagation Through Time (BPTT)",
                    "Extending backpropagation to sequential data")

    pw.text(dedent("""\
    When the computation involves a sequence of timesteps \u2014 as in recurrent neural networks \
    or ODE integrators \u2014 we must propagate gradients not just through layers but also \
    through time. This is Backpropagation Through Time (BPTT) (Werbos, 1990)."""))
    pw.gap()

    pw.text("4.1  The Recurrent Structure", fontsize=12, weight="bold")
    pw.text(dedent("""\
    A recurrent computation applies the same function f at each timestep, with shared \
    parameters \u03b8. The state s evolves as:"""))
    pw.equation(r"$\mathbf{s}_{t+1} = f(\mathbf{s}_t, \mathbf{u}_t;\; \boldsymbol{\theta})$")
    pw.text(dedent("""\
    where u_t is external input at time t. In an Euler-method neuron simulation, \u03b8 \
    are the biophysical parameters (g_L, E_L, C_m, ...), s_t = (v_t, w_t) is the neuron \
    state, and u_t = I_ext(t) is the injected current."""))
    pw.gap()

    pw.text("4.2  Unrolling the Computation", fontsize=12, weight="bold")
    pw.text(dedent("""\
    To apply backpropagation, we 'unroll' the recurrence into a feedforward graph with T \
    layers. Each layer shares the same parameters \u03b8. The loss may depend on the state at \
    every timestep (MSE loss) or only on derived features (spike count, timing):"""))
    pw.equation(r"$L = \mathcal{L}(\mathbf{s}_0, \mathbf{s}_1, \ldots, \mathbf{s}_T;\; \boldsymbol{\theta})$")

    pw.text("4.3  The BPTT Gradient", fontsize=12, weight="bold")
    pw.text(dedent("""\
    The total gradient decomposes into contributions from each timestep. Since \u03b8 is shared \
    across all steps, contributions accumulate (Goodfellow et al., 2016, Ch. 10.2):"""))
    pw.equation(r"$\frac{\partial L}{\partial \boldsymbol{\theta}} = \sum_{t=0}^{T-1} \frac{\partial L}{\partial \mathbf{s}_t} \cdot \frac{\partial \mathbf{s}_t}{\partial \boldsymbol{\theta}}$", fontsize=12)

    pw.text(dedent("""\
    The term \u2202L/\u2202s_t (the 'error signal' at time t) must be propagated backward from the \
    future. This involves the product of Jacobians through each step:"""))
    pw.equation(r"$\frac{\partial L}{\partial \mathbf{s}_t} = \frac{\partial L_t}{\partial \mathbf{s}_t} + \frac{\partial \mathbf{s}_{t+1}}{\partial \mathbf{s}_t}^\top \frac{\partial L}{\partial \mathbf{s}_{t+1}}$", fontsize=12)

    pw.text(dedent("""\
    where L_t is the per-step loss (if any) and the Jacobian \u2202s_{t+1}/\u2202s_t propagates \
    future error signals to earlier times. This recursive formula is the heart of BPTT."""))
    pw.gap()

    pw.text("4.4  Vanishing and Exploding Gradients", fontsize=12, weight="bold")
    pw.text(dedent("""\
    The error signal at time t depends on the product of T\u2212t Jacobians. If the spectral \
    radius of \u2202s_{t+1}/\u2202s_t is consistently < 1, gradients vanish exponentially; if > 1, \
    they explode. For the AdEx neuron with leak dynamics:"""))
    pw.equation(r"$\frac{\partial v_{t+1}}{\partial v_t} \approx 1 - \frac{\Delta t \cdot g_L}{C_m} \approx 0.9997$")
    pw.text(dedent("""\
    Over T = 5000 steps: 0.9997\u2075\u2070\u2070\u2070 \u2248 0.22. The gradient from the last timestep is \
    attenuated by 78% by the time it reaches the first timestep. For longer simulations or \
    higher leak, this factor approaches zero \u2014 the classic vanishing gradient problem \
    (Hochreiter, 1991; Bengio et al., 1994)."""))

    return pw.figures


def make_ch4_bptt_figure():
    """Figure: Unrolled computation graph with gradient flow."""
    fig, ax = plt.subplots(figsize=(A4_W, 3.5))
    fig.suptitle("Figure 4.1: Unrolled Computation Graph and Gradient Flow in BPTT",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-2, 2.5)
    ax.axis("off")

    # Timesteps
    T = 5
    xs = np.linspace(0.5, 6.5, T)

    for i, x in enumerate(xs):
        # State node
        c = plt.Circle((x, 1), 0.35, facecolor="#e3f2fd", edgecolor=COLOR_BLUE, lw=1.5)
        ax.add_patch(c)
        if i < T-1:
            ax.text(x, 1, f"$\\mathbf{{s}}_{i}$", ha="center", va="center", fontsize=10)
        else:
            ax.text(x, 1, f"$\\mathbf{{s}}_T$", ha="center", va="center", fontsize=10)

        # Input arrow from below
        ax.annotate("", xy=(x, 0.6), xytext=(x, -0.2),
                   arrowprops=dict(arrowstyle="->", color=COLOR_GREEN, lw=1))
        ax.text(x, -0.4, f"$\\mathbf{{u}}_{i}$" if i < T-1 else "$\\mathbf{u}_T$",
                ha="center", fontsize=8, color=COLOR_GREEN)

        # Forward arrow to next state
        if i < T-1:
            ax.annotate("", xy=(xs[i+1]-0.4, 1), xytext=(x+0.4, 1),
                       arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=1.5))
            mid = (x + xs[i+1]) / 2
            ax.text(mid, 1.5, r"$f(\cdot;\boldsymbol{\theta})$",
                    ha="center", fontsize=8, color=COLOR_BLUE)

    # Loss node
    ax.text(6.5, 2.2, "$L$", ha="center", fontsize=14, fontweight="bold",
            color=COLOR_ACCENT,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fce4ec", edgecolor=COLOR_ACCENT))

    # Backward gradient arrows (below)
    for i in range(T-1, 0, -1):
        x1, x2 = xs[i], xs[i-1]
        # Gradient decay
        alpha = 0.3 + 0.7 * (i / (T-1))
        ax.annotate("", xy=(x2+0.4, -0.8), xytext=(x1-0.4, -0.8),
                   arrowprops=dict(arrowstyle="->", color=COLOR_ACCENT, lw=1.5, alpha=alpha))

    # Labels
    ax.text(3.5, -1.3,
            r"$\frac{\partial L}{\partial \mathbf{s}_t} = \frac{\partial \mathbf{s}_{t+1}}{\partial \mathbf{s}_t}^\top \frac{\partial L}{\partial \mathbf{s}_{t+1}}$"
            "    (gradient decays backward through time)",
            ha="center", fontsize=9, color=COLOR_ACCENT)

    ax.text(0.0, 2.3, "Forward \u2192", color=COLOR_BLUE, fontsize=9, fontweight="bold")
    ax.text(0.0, -0.8, "\u2190 Backward", color=COLOR_ACCENT, fontsize=9, fontweight="bold")

    # Shared params annotation
    ax.text(3.5, 1.9, r"$\boldsymbol{\theta}$ shared across all steps $\Rightarrow$ gradients accumulate",
            ha="center", fontsize=9, fontstyle="italic", color="gray")

    fig.tight_layout()
    return fig


def make_vanishing_gradient_figure():
    """Figure: Gradient magnitude vs timestep."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(A4_W, 3.2))
    fig.suptitle("Figure 4.2: Vanishing Gradients Through Leak Dynamics",
                 fontsize=11, fontweight="bold", y=1.0)

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
    return fig


def make_ch4b_why_differentiable():
    """Page: Why the loss must be differentiable."""
    pw = PageWriter("4b. Why the Loss Function Must Be Differentiable",
                    "The mathematical requirement for gradient-based optimization")

    pw.text(dedent("""\
    Gradient descent requires computing the derivative of the loss with respect to every \
    trainable parameter. If any link in the computational chain from parameters to loss \
    is non-differentiable, the gradient is undefined at that point and the chain rule \
    cannot propagate through it. This section explains why this matters concretely."""))
    pw.gap()

    pw.text("4b.1  The Chain Rule Requires Derivatives Everywhere",
            fontsize=12, weight="bold")
    pw.text("Consider the chain from parameters to loss:")
    pw.equation(
        r"$\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial v_T} \cdot \frac{\partial v_T}{\partial v_{T-1}} \cdot \ldots \cdot \frac{\partial v_1}{\partial \theta}$")
    pw.text(dedent("""\
    This is a product of many factors. If any single factor is undefined (because the \
    function is not differentiable at the evaluation point), the entire product is \
    undefined. Gradient descent literally cannot compute a direction to move."""))
    pw.gap()

    pw.text("4b.2  Counterexample: A Non-Differentiable Loss",
            fontsize=12, weight="bold")
    pw.text(dedent("""\
    Suppose we define loss = 'number of spikes missed' (an integer). This is a step \
    function of the parameters: as you smoothly vary g_L, the spike count stays constant \
    (gradient = 0), then jumps by 1 at a critical threshold (gradient = undefined). The \
    optimizer sees a flat landscape everywhere except at invisible cliff edges:"""))
    pw.bullet("Gradient = 0: parameters don't change, training stalls")
    pw.bullet("At the cliff: gradient is infinite/undefined, step direction is meaningless")
    pw.bullet("No information about which direction to move or by how much")
    pw.gap()

    pw.text(dedent("""\
    Compare this to a differentiable approximation like soft spike count using a sigmoid: \
    as parameters change, the soft count changes continuously and its gradient points \
    toward the correct parameter region."""))
    pw.gap()

    pw.text("4b.3  What 'Differentiable' Means in Practice",
            fontsize=12, weight="bold")
    pw.text(dedent("""\
    A function is differentiable if small changes in its input produce proportionally \
    small, predictable changes in its output. For the loss function:"""))
    pw.bullet("Differentiable: L(theta + eps) = L(theta) + grad * eps + O(eps^2). "
              "The gradient tells us the linear approximation.")
    pw.bullet("Non-differentiable: L(theta + eps) could be anything. No linear "
              "approximation exists. Gradient descent has no useful signal.")
    pw.gap()
    pw.text(dedent("""\
    This is why every component in our pipeline must be differentiable: the AdEx dynamics \
    (solved via Euler steps, all smooth operations), the spike detection (made smooth via \
    surrogate gradients), the reset mechanism (continuous interpolation), and the loss \
    function itself (Baydin et al., 2018)."""))
    pw.gap()

    pw.text("4b.4  The Practical Consequence for Spiking Models",
            fontsize=12, weight="bold")
    pw.text(dedent("""\
    In spiking neuron models, the spike event is inherently non-differentiable (a \
    Heaviside step). This means that even if the loss function is perfectly smooth \
    (like MSE), the gradient still cannot flow through the spike. The chain is broken \
    at the spike, not at the loss. This is why surrogate gradients are necessary: they \
    restore differentiability at the spike event, making the entire chain differentiable \
    again. Without them, it does not matter how clever the loss function is \u2014 the \
    gradient will always be zero at spike events (Neftci et al., 2019)."""))

    return pw.figures


def make_ch4b_differentiability_figure():
    """Figure: Differentiable vs non-differentiable loss."""
    fig, axes = plt.subplots(1, 3, figsize=(A4_W, 3.2))
    fig.suptitle("Figure 4b.1: Why Differentiability Matters for Optimization",
                 fontsize=11, fontweight="bold", y=1.02)

    theta = np.linspace(-2, 2, 1000)

    # (a) Non-differentiable: integer spike count
    ax = axes[0]
    spike_count = np.floor(2.5 + 1.5 * np.tanh(theta * 2)).astype(int)
    ax.step(theta, spike_count, where="mid", color=COLOR_ACCENT, lw=2)
    ax.set_title("(a) Spike count (integer)", fontsize=9)
    ax.set_xlabel(r"Parameter $\theta$")
    ax.set_ylabel("Loss = |spikes missed|")
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.85, "Gradient = 0\neverywhere",
            transform=ax.transAxes, fontsize=8, ha="center",
            color=COLOR_ACCENT, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#fce4ec", alpha=0.8))

    # (b) Differentiable soft spike count
    ax = axes[1]
    soft_count = 2.5 + 1.5 * np.tanh(theta * 2)
    ax.plot(theta, soft_count, color=COLOR_GREEN, lw=2)
    # Show gradient arrows at a few points
    for t0 in [-1.0, 0.0, 0.8]:
        idx = np.argmin(np.abs(theta - t0))
        grad = 1.5 * 2 * (1 - np.tanh(t0 * 2)**2)
        ax.annotate("", xy=(t0 + 0.2, soft_count[idx] + grad * 0.2),
                    xytext=(t0, soft_count[idx]),
                    arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=1.5))
    ax.set_title("(b) Soft spike count", fontsize=9)
    ax.set_xlabel(r"Parameter $\theta$")
    ax.set_ylabel("Soft count (differentiable)")
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.85, "Gradient exists\neverywhere",
            transform=ax.transAxes, fontsize=8, ha="center",
            color=COLOR_GREEN, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#e8f5e9", alpha=0.8))

    # (c) The gradient signal comparison
    ax = axes[2]
    # Non-differentiable gradient (zero everywhere)
    ax.axhline(0, color=COLOR_ACCENT, lw=2, label="Hard count: grad=0", ls="--")
    # Differentiable gradient
    soft_grad = 1.5 * 2 * (1 - np.tanh(theta * 2)**2)
    ax.plot(theta, soft_grad, color=COLOR_GREEN, lw=2, label="Soft count: smooth")
    ax.set_title("(c) Gradient comparison", fontsize=9)
    ax.set_xlabel(r"Parameter $\theta$")
    ax.set_ylabel(r"$\partial L / \partial \theta$")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def make_ch4c_bptt_detailed():
    """Page: Detailed BPTT walkthrough for AdEx."""
    pw = PageWriter("4c. BPTT in Detail: From Loss to Parameter Update",
                                "Tracing the gradient through a neuron simulation")

    pw.text(dedent("""\
    Let us trace, step by step, how a scalar loss value propagates backward through the \
    simulation to produce a gradient for a single parameter. We use g_L (leak conductance) \
    as our running example."""))
    pw.gap()

    pw.text("4c.1  Forward: How g_L Affects the Voltage", fontsize=12, weight="bold")
    pw.text(dedent("""\
    At each timestep, the membrane voltage evolves according to the AdEx ODE:"""))
    pw.equation(
        r"$v_{t+1} = v_t + \frac{\Delta t}{C_m}\left[g_L(E_L - v_t) + g_L \Delta_T e^{\frac{v_t - v_T}{\Delta_T}} - w_t + I_t\right]$",
        fontsize=11)
    pw.text(dedent("""\
    The parameter g_L appears explicitly in the leak term g_L(E_L - v_t) and in the \
    exponential term. A larger g_L pulls v_t more strongly toward E_L (the resting \
    potential) and also amplifies the exponential spike initiation. Every single \
    timestep's voltage depends on g_L."""))
    pw.gap()

    pw.text("4c.2  How the Loss 'Sees' the Voltage", fontsize=12, weight="bold")
    pw.text(dedent("""\
    The loss function takes the entire voltage trace [v_0, v_1, ..., v_T] and compresses \
    it to a single number. Different loss functions do this differently:"""))
    pw.gap()
    pw.bullet("MSE: averages (v_t - v_target_t)^2 over all T timesteps. Every "
                   "timestep contributes equally to the loss.")
    pw.bullet("Van Rossum: convolves the soft spike indicators s_t with an "
                   "exponential kernel, then computes L2 distance. Only timesteps near "
                   "spikes contribute significantly.")
    pw.bullet("Deistler: computes mean and std of voltage in windows. All "
                   "timesteps within each window contribute equally.")
    pw.gap()

    pw.text("4c.3  Backward: The Gradient Accumulates Over Time",
                 fontsize=12, weight="bold")
    pw.text(dedent("""\
    The total gradient is a sum over all timesteps (since g_L is used at every step):"""))
    pw.equation(
        r"$\frac{\partial L}{\partial g_L} = \sum_{t=0}^{T-1} \frac{\partial L}{\partial v_t} \cdot \frac{\partial v_t}{\partial g_L}$",
        fontsize=12)
    pw.text(dedent("""\
    Each term in this sum has two parts:"""))
    pw.gap()

    pw.text("Part A: How much does the loss care about v_t?", weight="bold",
                 indent=0.02)
    pw.text(dedent("""\
    This is the 'error signal' at time t. For MSE: (2/T)(v_t - v_target_t). For Van \
    Rossum: non-zero only near spikes. This is where the loss function choice matters \
    most \u2014 it determines which timesteps send gradient signal backward."""), indent=0.04)
    pw.gap()

    pw.text("Part B: How much does g_L affect v_t?", weight="bold",
                 indent=0.02)
    pw.text(dedent("""\
    This requires unrolling the chain rule through all preceding timesteps. The direct \
    effect of g_L on v_t is (dt/C_m)(E_L - v_t), but g_L also affected v_{t-1}, which \
    affected v_t, and so on. The total effect is a sum of direct and indirect paths:"""),
                 indent=0.04)
    pw.equation(
        r"$\frac{\partial v_t}{\partial g_L} = \frac{\partial v_t}{\partial v_{t-1}} \cdot \frac{\partial v_{t-1}}{\partial g_L} + \frac{\Delta t}{C_m}(E_L - v_{t-1} + \ldots)$",
        fontsize=11)
    pw.text(dedent("""\
    The first term is the indirect effect (past influence, decaying through leak), \
    and the second term is the direct effect at this timestep. The indirect term \
    is multiplied by the Jacobian (approx. 0.9997 per step), causing exponential decay \
    of information from earlier timesteps."""), indent=0.04)
    pw.gap()

    pw.text("4c.4  The Final Update", fontsize=12, weight="bold")
    pw.text(dedent("""\
    After summing over all T timesteps, we have a single number: the total gradient \
    of L with respect to g_L. This tells the optimizer: 'increasing g_L by a tiny amount \
    epsilon would change L by approximately (gradient * epsilon).' The Adam optimizer then \
    adjusts g_L by an amount scaled by the learning rate and its internal momentum."""))

    return pw.figures


def make_ch4c_gradient_path_figure():
    """Figure: Tracing the gradient path for a single parameter."""
    fig, ax = plt.subplots(figsize=(A4_W, 4.5))
    fig.suptitle(
        "Figure 4c.1: Gradient Paths from Loss to a Single Parameter",
        fontsize=11, fontweight="bold", y=0.99)
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-3, 4)
    ax.axis("off")

    # Timeline of timesteps
    T = 6
    xs = np.linspace(0.5, 8.5, T)
    vy = 2.5  # voltage node y

    # Parameter node (bottom, shared)
    param_y = -1.5
    ax.text(4.5, param_y, r"$g_L$", fontsize=14, ha="center", va="center",
            fontweight="bold", color=COLOR_GREEN,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9",
                     edgecolor=COLOR_GREEN, lw=2))

    # Loss node
    loss_x = 9.5
    ax.text(loss_x, vy, "$L$", fontsize=14, ha="center", va="center",
            fontweight="bold", color=COLOR_ACCENT,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fce4ec",
                     edgecolor=COLOR_ACCENT, lw=2))

    for i, x in enumerate(xs):
        # Voltage node
        c = plt.Circle((x, vy), 0.3, facecolor="#e3f2fd",
                       edgecolor=COLOR_BLUE, lw=1.5)
        ax.add_patch(c)
        ax.text(x, vy, f"$v_{i}$", ha="center", va="center", fontsize=9)

        # Forward arrow to next voltage
        if i < T-1:
            ax.annotate("", xy=(xs[i+1]-0.35, vy), xytext=(x+0.35, vy),
                       arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=1.2))

        # Arrow from parameter to each voltage (direct influence)
        alpha = 0.15 + 0.85 * (i / (T-1))
        ax.annotate("", xy=(x, vy - 0.35), xytext=(4.5 + (x-4.5)*0.15, param_y + 0.3),
                   arrowprops=dict(arrowstyle="->", color=COLOR_GREEN, lw=1,
                                  alpha=alpha, connectionstyle="arc3,rad=0.2"))

    # Arrow from last voltage to loss
    ax.annotate("", xy=(loss_x - 0.3, vy), xytext=(xs[-1]+0.35, vy),
               arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=1.2))

    # Backward arrows (gradient flow)
    bk_y = 1.2
    for i in range(T-1, -1, -1):
        alpha_bk = 0.2 + 0.8 * (i / (T-1))
        if i == T-1:
            # From loss
            ax.annotate("", xy=(xs[i]+0.1, bk_y+0.3), xytext=(loss_x - 0.3, vy - 0.4),
                       arrowprops=dict(arrowstyle="->", color=COLOR_ACCENT,
                                      lw=1.5, alpha=alpha_bk))
        if i > 0:
            # Between timesteps
            ax.annotate("", xy=(xs[i-1]+0.3, bk_y), xytext=(xs[i]-0.3, bk_y),
                       arrowprops=dict(arrowstyle="->", color=COLOR_ACCENT,
                                      lw=1.5, alpha=alpha_bk))

    # Gradient accumulation arrows (from each timestep down to param)
    for i, x in enumerate(xs):
        alpha_acc = 0.15 + 0.85 * (i / (T-1))
        ax.annotate("", xy=(4.5 + (x-4.5)*0.1, param_y + 0.4),
                   xytext=(x, bk_y - 0.15),
                   arrowprops=dict(arrowstyle="->", color=COLOR_ORANGE,
                                  lw=1, alpha=alpha_acc, ls="--",
                                  connectionstyle="arc3,rad=-0.15"))

    # Labels
    ax.text(4.5, 3.7, "Forward: parameter influences every timestep",
            ha="center", fontsize=9, color=COLOR_BLUE, fontstyle="italic")
    ax.text(4.5, 0.5,
            "Backward: error signal flows back and accumulates at parameter",
            ha="center", fontsize=9, color=COLOR_ACCENT, fontstyle="italic")

    # Legend
    legend_y = -2.5
    for color, label, xpos in [
        (COLOR_GREEN, "Direct influence (forward)", 1.5),
        (COLOR_ACCENT, "Error signal (backward)", 4.5),
        (COLOR_ORANGE, "Gradient accumulation", 7.5),
    ]:
        ax.plot([xpos-0.3, xpos+0.3], [legend_y, legend_y], color=color, lw=2)
        ax.text(xpos+0.5, legend_y, label, fontsize=8, va="center", color=color)

    fig.tight_layout()
    return fig


def make_ch4d_spike_gated_gradients():
    """Page: Gradient existence around spikes."""
    pw = PageWriter(
        "4d. Spike-Gated Gradients",
        "Where gradients exist and where they vanish in a voltage trace")

    pw.text(dedent("""\
    Not all parameters receive gradient signal at every timestep. The surrogate gradient \
    creates a 'gate' that only opens when the membrane potential is close to threshold. \
    This has profound consequences for which parameters can be trained and when."""))
    pw.gap()

    pw.text("4d.1  Two Types of Gradient Pathways", fontsize=12, weight="bold")
    pw.text(dedent("""\
    There are two distinct pathways through which a parameter gradient can flow:"""))
    pw.gap()

    pw.text("Pathway 1: Through the subthreshold dynamics",
                 weight="bold", indent=0.02)
    pw.text(dedent("""\
    Parameters that appear in the ODE (g_L, E_L, C_m, a, tau_w) influence v_t through \
    the smooth Euler update. Their gradient exists at every timestep, flowing through \
    dv/dt = (leak + exponential - w + I) / C_m. This pathway is always active but \
    subject to exponential decay through the leak dynamics."""), indent=0.04)
    pw.gap()

    pw.text("Pathway 2: Through the spike event",
                 weight="bold", indent=0.02)
    pw.text(dedent("""\
    Parameters involved in the reset mechanism (v_reset, v_threshold, b) primarily \
    influence the voltage through the spike equation: v_new = s * v_reset + (1-s) * v. \
    The gradient for these parameters requires the surrogate gradient ds/dv to be \
    non-zero. This only happens when v is within the surrogate gradient window \
    around v_threshold (approximately +/- 1-2 mV for beta = 5-10)."""), indent=0.04)
    pw.gap()

    pw.text("4d.2  When Are Gradients Non-Zero?", fontsize=12, weight="bold")
    pw.text(dedent("""\
    Consider a simulation with 5000 timesteps and 3 spikes. The surrogate gradient is \
    non-zero in a narrow window around each spike. With beta=10 and dt=0.025ms, the \
    window spans roughly 80 timesteps (2 mV / (dv/dt * dt)). That means:"""))
    pw.bullet("Subthreshold parameters (g_L, E_L, C_m): gradient exists at "
                   "all 5000 timesteps, but decays exponentially backward. Dominated "
                   "by the last ~1000 timesteps.")
    pw.bullet("Spike parameters (v_reset, b): gradient exists at ~240 "
                   "timesteps (3 spikes x 80 timesteps each). That is only 4.8% of "
                   "the trace. The gradient is zero for 95% of the simulation.")
    pw.bullet("Threshold (v_threshold): gradient at ~240 timesteps via "
                   "the surrogate, plus a weak contribution through the exponential "
                   "term at all timesteps.")
    pw.gap()

    pw.text("4d.3  The Consequence: Sparse Gradient Signal",
                 fontsize=12, weight="bold")
    pw.text(dedent("""\
    This sparsity creates an asymmetry: subthreshold parameters receive continuous but \
    decaying gradient, while spike parameters receive rare but potentially strong bursts. \
    If the model is not spiking at all (a common failure mode), then the surrogate window \
    is never entered, and spike parameters receive exactly zero gradient \u2014 they cannot \
    be updated regardless of how large the loss is."""))
    pw.gap()

    pw.text(dedent("""\
    This is why the loss function must provide signal even in non-spiking regimes. A pure \
    spike-based loss (like Van Rossum distance) gives zero loss and zero gradient when \
    neither simulation nor target spikes exist in a region. Combining it with a \
    subthreshold voltage component ensures that there is always some gradient to pull \
    the dynamics closer to threshold (Zenke & Ganguli, 2018)."""))
    pw.gap()

    pw.text("4d.4  Parameter-Specific Gradient Expressions",
                 fontsize=12, weight="bold")
    pw.text(dedent("""\
    The local gradient of v_{t+1} with respect to each parameter (at a single step):"""))
    pw.gap()
    pw.bullet("g_L:  (dt/C_m) * (E_L - v_t + delta_T * exp(...))     [always non-zero]",
                   indent=0.02)
    pw.bullet("E_L:  (dt/C_m) * g_L                                   [always non-zero]",
                   indent=0.02)
    pw.bullet("C_m:  -(dt/C_m^2) * (entire drift term)               [always non-zero]",
                   indent=0.02)
    pw.bullet("v_reset: s_t (surrogate spike indicator)                [only near spikes]",
                   indent=0.02)
    pw.bullet("b:  s_t (via adaptation update w += s*b)                [only near spikes]",
                   indent=0.02)

    return pw.figures


def make_ch4d_gradient_existence_figure():
    """Figure: Where gradients exist on a voltage trace."""
    fig, axes = plt.subplots(4, 1, figsize=(A4_W, 7.5), sharex=True)
    fig.suptitle(
        "Figure 4d.1: Gradient Existence Map Across a Voltage Trace",
        fontsize=11, fontweight="bold", y=1.0)

    np.random.seed(42)
    T = 2000
    dt = 0.025
    t = np.arange(T) * dt

    # Simulate a simple voltage trace with spikes
    v = np.full(T, -65.0)
    v_threshold = 0.0
    # Subthreshold oscillations rising toward threshold
    base = -65 + 15 * (t / t[-1])  # slowly depolarizing
    osc = 5 * np.sin(2 * np.pi * t / 8)  # oscillations
    v = base + osc + np.random.randn(T) * 0.5

    # Add spikes
    spike_centers = [600, 1200, 1700]
    spike_indicators = np.zeros(T)
    for sc in spike_centers:
        # Approach threshold, cross, reset
        for j in range(max(0,sc-40), min(T,sc)):
            v[j] = v[j] + (v_threshold - v[j]) * ((j - (sc-40)) / 40.0) * 1.2
        if sc < T:
            v[sc] = 20  # spike peak
        if sc+1 < T:
            v[sc+1] = -55  # reset
        for j in range(max(0,sc+2), min(T, sc+20)):
            v[j] = -55 + (v[j] - (-55)) * 0.3  # recovery

    # Compute surrogate gradient value at each timestep
    beta = 10.0
    x_surr = v - v_threshold
    sig = 1.0 / (1.0 + np.exp(-beta * x_surr))
    surrogate_grad = beta * sig * (1 - sig)

    # (a) Voltage trace
    ax = axes[0]
    ax.plot(t, v, color=COLOR_PRIMARY, lw=0.8)
    ax.axhline(v_threshold, color=COLOR_ACCENT, ls="--", lw=1, alpha=0.7,
               label="$v_{\\mathrm{thresh}}$")
    ax.set_ylabel("v (mV)")
    ax.set_title("(a) Membrane potential trace", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(-75, 25)

    # Shade surrogate window
    window_low = v_threshold - 2
    window_high = v_threshold + 2
    ax.axhspan(window_low, window_high, alpha=0.15, color=COLOR_ORANGE,
               label="Surrogate window")

    # (b) Surrogate gradient magnitude
    ax = axes[1]
    ax.fill_between(t, surrogate_grad, alpha=0.5, color=COLOR_ORANGE)
    ax.plot(t, surrogate_grad, color=COLOR_ORANGE, lw=0.5)
    ax.set_ylabel("Surrogate\ngradient")
    ax.set_title("(b) Surrogate gradient ds/dv: non-zero only near threshold",
                 fontsize=10, loc="left")

    # (c) Gradient existence for subthreshold param (g_L) - always present
    ax = axes[2]
    # Gradient magnitude decays backward from end
    factor = 0.9997
    gl_grad = np.zeros(T)
    for i in range(T):
        gl_grad[i] = abs(dt / 200 * (-65 - v[i]))  # local contribution
    # Backward-weighted
    weights = factor ** np.arange(T)[::-1]
    gl_grad_weighted = gl_grad * weights
    gl_grad_weighted /= gl_grad_weighted.max() + 1e-10

    ax.fill_between(t, gl_grad_weighted, alpha=0.4, color=COLOR_BLUE)
    ax.plot(t, gl_grad_weighted, color=COLOR_BLUE, lw=0.5)
    ax.set_ylabel("Gradient\ncontribution")
    ax.set_title("(c) g_L gradient: continuous, grows toward end of trace",
                 fontsize=10, loc="left")

    # (d) Gradient existence for spike param (v_reset) - only near spikes
    ax = axes[3]
    vreset_grad = surrogate_grad.copy()
    # Only non-zero near spikes, multiplied by backward decay
    vreset_grad_weighted = vreset_grad * weights
    vreset_grad_weighted /= vreset_grad_weighted.max() + 1e-10

    ax.fill_between(t, vreset_grad_weighted, alpha=0.4, color=COLOR_ACCENT)
    ax.plot(t, vreset_grad_weighted, color=COLOR_ACCENT, lw=0.5)
    ax.set_ylabel("Gradient\ncontribution")
    ax.set_title("(d) v_reset gradient: only exists in narrow windows around spikes",
                 fontsize=10, loc="left")
    ax.set_xlabel("Time (ms)")

    # Annotate spike windows
    for sc in spike_centers:
        tc = sc * dt
        for a in [axes[1], axes[3]]:
            a.axvline(tc, color="gray", ls=":", lw=0.5, alpha=0.5)

    fig.tight_layout()
    return fig


def make_ch4e_loss_to_param():
    """Page: How loss connects to parameters through voltage."""
    pw = PageWriter(
        "4e. From Loss to Parameters: The Complete Chain",
        "A concrete walkthrough for MSE and Van Rossum")

    pw.text(dedent("""\
    To make the connection between loss function and parameter update fully concrete, let \
    us trace the gradient chain for two loss functions applied to the same trace."""))
    pw.gap()

    pw.text("4e.1  MSE Loss: Every Timestep Speaks", fontsize=12, weight="bold")
    pw.text(dedent("""\
    The MSE loss produces a gradient at every single timestep:"""))
    pw.equation(
        r"$\frac{\partial L_\mathrm{MSE}}{\partial v_t} = \frac{2}{T}(v_t - v_t^\mathrm{target})$",
        fontsize=12)
    pw.text(dedent("""\
    This error signal then travels backward through time. For a subthreshold parameter \
    like g_L, the gradient at each timestep is non-zero, so the contributions from all T \
    timesteps accumulate. The total gradient is dominated by late timesteps (less decay) \
    and by timesteps where the voltage error is large."""))
    pw.gap()
    pw.text(dedent("""\
    For a spike parameter like v_reset, the gradient requires the surrogate to be non-zero. \
    Even though MSE provides error signal at every timestep, the gradient for v_reset is \
    zero at timesteps where the voltage is far from threshold. Only when v approaches \
    v_threshold does the surrogate 'open the gate' and allow gradient to flow to v_reset. \
    The resulting gradient is the product of:"""))
    pw.equation(
        r"$\frac{\partial L}{\partial v_\mathrm{reset}} = \sum_{t\, :\, |v_t - v_\mathrm{thresh}| < \epsilon} \frac{\partial L}{\partial v_t} \cdot s_t$",
        fontsize=11)
    pw.text(dedent("""\
    where the sum is effectively restricted to the few timesteps near threshold crossings. \
    The term s_t is the surrogate spike indicator, which gates the gradient flow."""))
    pw.gap()

    pw.text("4e.2  Van Rossum: Only Spikes Matter", fontsize=12, weight="bold")
    pw.text(dedent("""\
    The Van Rossum distance operates on spike trains, not raw voltage. Its gradient path is:"""))
    pw.equation(
        r"$L_\mathrm{VR} \;\rightarrow\; \mathrm{filtered\;spike\;train} \;\rightarrow\; s_t \;\rightarrow\; v_t \;\rightarrow\; \theta$",
        fontsize=11)
    pw.text(dedent("""\
    The critical difference: the Van Rossum loss produces gradient at s_t (the soft spike \
    indicator), not at v_t (the voltage). The exponential kernel in the convolution smears \
    this gradient over a temporal window of width tau, creating a smooth 'pull' toward the \
    correct spike time. A spike that is 5ms too early creates a gradient that says 'delay \
    this spike' over the entire 5ms window, rather than a point error."""))
    pw.gap()

    pw.text(dedent("""\
    However, the gradient from s_t to v_t still requires the surrogate gate. And the \
    gradient from v_t to theta still requires backpropagation through time with its \
    exponential decay. The Van Rossum loss improves the 'starting signal' at the loss end, \
    but does not bypass the spike or temporal decay challenges."""))
    pw.gap()

    pw.text("4e.3  The Fundamental Trade-off", fontsize=12, weight="bold")
    pw.text(dedent("""\
    Each loss function makes a trade-off in what information it provides:"""))
    pw.gap()
    pw.bullet("MSE: gradient everywhere, but misleading near spikes (pushes "
                   "toward non-spiking). Subthreshold params get good signal; spike "
                   "params get bad signal.")
    pw.bullet("Van Rossum: gradient only near spikes, but informative and "
                   "temporally smooth. Spike params get good signal; subthreshold "
                   "params get no signal (without a voltage component).")
    pw.bullet("Deistler: gradient everywhere but uniform \u2014 no temporal "
                   "structure. All params get weak, unspecific signal.")
    pw.gap()
    pw.text(dedent("""\
    This motivates composite losses that combine a spike-aware component (Van Rossum) with \
    a subthreshold component (voltage MAE below a clamp), providing gradient signal through \
    both pathways simultaneously."""))

    return pw.figures


def make_ch5_page1():
    """Chapter 5: The Spike Discontinuity Problem."""
    fig, ax, y = new_text_page("5. The Spike Discontinuity Problem",
                                "Why standard backpropagation fails for spiking neurons")

    y = add_text(ax, y, dedent("""\
    The Adaptive Exponential Integrate-and-Fire (AdEx) model (Brette & Gerstner, 2005) \
    uses a discrete reset mechanism: when membrane potential v crosses a threshold \
    v_threshold, v is instantaneously reset to v_reset. This is modeled by the Heaviside \
    step function H(x), where H(x) = 1 if x >= 0 and H(x) = 0 otherwise:"""))
    y = add_equation(ax, y, r"$\mathrm{spike}_t = H(v_t - v_\mathrm{thresh})$")

    y = add_text(ax, y, "5.1  The Zero Gradient Problem", fontsize=12, weight="bold")
    y = add_text(ax, y, dedent("""\
    The Heaviside function is not differentiable at x=0, and its derivative is zero \
    everywhere else:"""))
    y = add_equation(ax, y, r"$H'(x) = 0 \quad \forall\; x \neq 0$")
    y = add_text(ax, y, dedent("""\
    When we try to backpropagate through a spike event, the gradient encounters this zero \
    derivative. The chain rule multiplies by zero, and all upstream gradients vanish. \
    Parameters that influence spiking behavior (v_threshold, v_reset, b) receive zero \
    gradient signal \u2014 the optimizer cannot learn how to adjust them \
    (Neftci et al., 2019)."""))
    y -= 0.005

    y = add_text(ax, y, "5.2  The Reset Mechanism in Detail", fontsize=12, weight="bold")
    y = add_text(ax, y, "Standard (hard) reset:", weight="bold", indent=0.02)
    y = add_text(ax, y, dedent("""\
    If v_t >= v_thresh:  v_{t+1} = v_reset  (hard conditional)
    Otherwise:           v_{t+1} = v_t + dt * f(v_t, w_t, I_t; theta)"""), indent=0.04, fontsize=9)
    y = add_text(ax, y, dedent("""\
    This conditional assignment is a step function in disguise: the derivative \
    \u2202v_{t+1}/\u2202v_t is undefined at the threshold and the reset branch has zero \
    contribution from the pre-reset voltage."""))
    y -= 0.005

    y = add_text(ax, y, "Continuous (soft) reset:", weight="bold", indent=0.02)
    y = add_equation(ax, y, r"$v_{t+1} = s_t \cdot v_\mathrm{reset} + (1 - s_t) \cdot \tilde{v}_t$")
    y = add_text(ax, y, dedent("""\
    where s_t \u2208 [0,1] is a smooth spike indicator and v\u0303_t is the pre-reset voltage. This \
    formulation is differentiable everywhere via the product rule \u2014 but requires s_t to \
    have a non-zero gradient, which is the role of surrogate gradients."""))
    y -= 0.005

    y = add_text(ax, y, "5.3  Impact on the Loss Landscape", fontsize=12, weight="bold")
    y = add_text(ax, y, dedent("""\
    The spike threshold creates a bifurcation in the dynamics: small changes in parameters \
    can cause spikes to appear or disappear. This creates a highly non-convex loss landscape \
    with sharp cliffs and flat plateaus. Without gradients through spike events, the \
    optimizer is blind to this structure (Zenke & Ganguli, 2018)."""))

    return fig


def make_heaviside_figure():
    """Figure: Heaviside function and its derivative."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(A4_W, 3))
    fig.suptitle("Figure 5.1: The Heaviside Function and Its (Non-)Derivative",
                 fontsize=11, fontweight="bold", y=1.02)

    x = np.linspace(-3, 3, 1000)

    # Left: Heaviside
    H = np.where(x >= 0, 1.0, 0.0)
    ax1.plot(x, H, color=COLOR_PRIMARY, lw=2.5, label="$H(x)$")
    ax1.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax1.scatter([0, 0], [0, 1], color=COLOR_PRIMARY, zorder=5, s=40)
    ax1.set_xlabel("$x = v - v_{\\mathrm{thresh}}$")
    ax1.set_ylabel("$H(x)$")
    ax1.set_title("(a) Heaviside step function", fontsize=10)
    ax1.set_ylim(-0.2, 1.3)
    ax1.grid(True, alpha=0.3)
    ax1.annotate("No spike", xy=(-1.5, 0.1), fontsize=9, color=COLOR_BLUE)
    ax1.annotate("Spike!", xy=(0.5, 0.85), fontsize=9, color=COLOR_ACCENT)

    # Right: Derivative
    ax2.axhline(0, color=COLOR_PRIMARY, lw=2.5, label="$H'(x) = 0$ everywhere")
    ax2.annotate("", xy=(0, 2.5), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=COLOR_ACCENT, lw=2.5))
    ax2.text(0.3, 2.2, "$\\delta(0)$\n(undefined)", fontsize=9, color=COLOR_ACCENT)
    ax2.set_xlabel("$x = v - v_{\\mathrm{thresh}}$")
    ax2.set_ylabel("$H'(x)$")
    ax2.set_title("(b) Derivative: zero almost everywhere", fontsize=10)
    ax2.set_ylim(-0.5, 3)
    ax2.set_xlim(-3, 3)
    ax2.grid(True, alpha=0.3)
    ax2.text(-2.5, 1.5, "Gradient = 0\n\u2192 No learning signal",
             fontsize=9, color=COLOR_ACCENT,
             bbox=dict(boxstyle="round", facecolor="#fce4ec", edgecolor=COLOR_ACCENT, alpha=0.8))

    fig.tight_layout()
    return fig


def make_ch6_page1():
    """Chapter 6: Surrogate Gradients."""
    pw = PageWriter("6. Surrogate Gradient Methods",
                                "Replacing H'(x) with smooth approximations")

    pw.text(dedent("""\
    Surrogate gradient methods (Neftci et al., 2019; Zenke & Ganguli, 2018) resolve the \
    zero-gradient problem by using a smooth function in the backward pass only. The forward \
    pass retains the hard threshold for accurate spiking dynamics. This is sometimes called \
    the 'straight-through estimator' generalization (Bengio et al., 2013)."""))
    pw.gap()

    pw.text("6.1  The Core Idea", fontsize=12, weight="bold")
    pw.text(dedent("""\
    Define a function with decoupled forward and backward passes:"""))
    pw.equation(r"$\text{Forward:} \quad s = H(v - v_\mathrm{thresh})$")
    pw.equation(r"$\text{Backward:} \quad \frac{\partial s}{\partial v} \approx \tilde{\sigma}'(v - v_\mathrm{thresh})$")
    pw.text(dedent("""\
    where \u03c3\u0303' is a smooth surrogate derivative. In JAX, this is implemented via \
    custom_vjp, which lets us specify an arbitrary VJP rule for any function."""))
    pw.gap()

    pw.text("6.2  Three Common Surrogate Functions", fontsize=12, weight="bold")
    pw.gap()

    pw.text("Sigmoid surrogate:", weight="bold", indent=0.02)
    pw.equation(r"$\tilde{\sigma}'(x) = \beta \cdot \sigma(\beta x) \cdot (1 - \sigma(\beta x))$")
    pw.text(dedent("""\
    The derivative of the sigmoid function scaled by \u03b2. Bell-shaped, smooth, and \
    concentrated around x=0. Peak gradient: \u03b2/4."""), indent=0.04)
    pw.gap()

    pw.text("Exponential surrogate:", weight="bold", indent=0.02)
    pw.equation(r"$\tilde{\sigma}'(x) = \beta \cdot \exp(-\beta |x|)$")
    pw.text(dedent("""\
    Symmetric exponential decay. Sharper peak than sigmoid but faster decay. Commonly \
    used in SNN literature."""), indent=0.04)
    pw.gap()

    pw.text("SuperSpike surrogate (Zenke & Ganguli, 2018):", weight="bold", indent=0.02)
    pw.equation(r"$\tilde{\sigma}'(x) = \frac{1}{(\beta |x| + 1)^2}$")
    pw.text(dedent("""\
    Polynomial (algebraic) decay with heavier tails than either sigmoid or exponential. \
    Provides non-zero gradient signal over a wider voltage range, helping 'rescue' \
    neurons that are far from threshold."""), indent=0.04)
    pw.gap()

    pw.text("6.3  The \u03b2 Parameter: Width vs. Accuracy Trade-off", fontsize=12, weight="bold")
    pw.text(dedent("""\
    The slope parameter \u03b2 controls the sharpness of the surrogate:"""))
    pw.bullet("High \u03b2 (e.g., 25): Narrow gradient window (~0.4 mV). Accurate approximation "
                   "of the Heaviside, but most voltage values produce zero gradient. Risk of 'dead neurons' "
                   "that cannot learn to spike.")
    pw.bullet("Low \u03b2 (e.g., 5): Wide gradient window (~2 mV). Biased approximation, but "
                   "informative gradients over a larger range. Easier optimization.")
    pw.gap()
    pw.text(dedent("""\
    Gygax & Zenke (2025) analyze this trade-off theoretically and recommend \u03b2 \u2208 [5, 10] \
    as a practical compromise. They show that the 'dead neuron' problem (analogous to \
    dying ReLU) is more harmful than the bias from a wider surrogate."""))

    return pw.figures


def make_surrogate_comparison_figure():
    """Figure: Comparison of three surrogate gradient functions."""
    fig, axes = plt.subplots(2, 2, figsize=(A4_W, 6))
    fig.suptitle("Figure 6.1: Surrogate Gradient Functions and Their \u03b2-Dependence",
                 fontsize=11, fontweight="bold", y=1.0)

    x = np.linspace(-5, 5, 1000)

    # (a) All three at same beta
    ax = axes[0, 0]
    beta = 10.0
    sigmoid_grad = beta * (1 / (1 + np.exp(-beta * x))) * (1 - 1 / (1 + np.exp(-beta * x)))
    exp_grad = beta * np.exp(-beta * np.abs(x))
    superspike_grad = 1.0 / (beta * np.abs(x) + 1)**2

    ax.plot(x, sigmoid_grad, label="Sigmoid", color=COLOR_BLUE, lw=1.8)
    ax.plot(x, exp_grad, label="Exponential", color=COLOR_GREEN, lw=1.8)
    ax.plot(x, superspike_grad, label="SuperSpike", color=COLOR_ACCENT, lw=1.8)
    ax.set_title(f"(a) All surrogates, \u03b2={beta:.0f}", fontsize=10)
    ax.set_xlabel("$x = v - v_{\\mathrm{thresh}}$ (mV)")
    ax.set_ylabel("Surrogate gradient")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)

    # (b) Sigmoid at different betas
    ax = axes[0, 1]
    for beta, ls in [(5, "-"), (10, "--"), (25, "-.")]:
        sig = 1 / (1 + np.exp(-beta * x))
        grad = beta * sig * (1 - sig)
        ax.plot(x, grad, label=f"\u03b2={beta}", lw=1.5, ls=ls, color=COLOR_BLUE)
    ax.set_title("(b) Sigmoid: effect of \u03b2", fontsize=10)
    ax.set_xlabel("$x$ (mV)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)

    # (c) Log-scale tails comparison
    ax = axes[1, 0]
    x_pos = np.linspace(0.01, 5, 500)
    beta = 10.0
    sig_pos = beta * (1/(1+np.exp(-beta*x_pos))) * (1 - 1/(1+np.exp(-beta*x_pos)))
    exp_pos = beta * np.exp(-beta * x_pos)
    ss_pos = 1.0 / (beta * x_pos + 1)**2

    ax.semilogy(x_pos, sig_pos, label="Sigmoid", color=COLOR_BLUE, lw=1.8)
    ax.semilogy(x_pos, exp_pos, label="Exponential", color=COLOR_GREEN, lw=1.8)
    ax.semilogy(x_pos, ss_pos, label="SuperSpike", color=COLOR_ACCENT, lw=1.8)
    ax.set_title("(c) Tail behavior (log scale), \u03b2=10", fontsize=10)
    ax.set_xlabel("Distance from threshold (mV)")
    ax.set_ylabel("Gradient magnitude (log)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-8, 1e1)
    ax.axhline(1e-6, color="gray", ls=":", lw=0.8)
    ax.text(3.5, 2e-6, "effectively zero", fontsize=7, color="gray", ha="center")

    # (d) Effective window width vs beta
    ax = axes[1, 1]
    betas = np.linspace(1, 30, 100)
    # Define "window" as range where gradient > 1% of peak
    # Sigmoid peak = beta/4, so window where beta*sig*(1-sig) > 0.01*beta/4
    # Numerically: find x where grad = 0.01 * peak
    widths_sig = []
    widths_exp = []
    widths_ss = []
    for b in betas:
        # Sigmoid: peak = b/4, width where grad > 1% of peak
        # beta * sig*(1-sig) > 0.01 * beta/4 → sig*(1-sig) > 0.0025
        # sig*(1-sig) = 0.0025 → sig = 0.9975 or 0.0025 → x = ln(0.9975/0.0025)/b ≈ ln(399)/b
        w = 2 * np.log(399) / b
        widths_sig.append(w)
        # Exponential: beta*exp(-beta*x) > 0.01*beta → exp(-beta*x) > 0.01 → x < ln(100)/beta
        w = 2 * np.log(100) / b
        widths_exp.append(w)
        # SuperSpike: 1/(b*x+1)^2 > 0.01*1 → (b*x+1)^2 < 100 → b*x < 9 → x < 9/b
        w = 2 * 9 / b
        widths_ss.append(w)

    ax.plot(betas, widths_sig, label="Sigmoid", color=COLOR_BLUE, lw=1.8)
    ax.plot(betas, widths_exp, label="Exponential", color=COLOR_GREEN, lw=1.8)
    ax.plot(betas, widths_ss, label="SuperSpike", color=COLOR_ACCENT, lw=1.8)
    ax.set_title("(d) Effective gradient window width", fontsize=10)
    ax.set_xlabel("\u03b2 (surrogate slope)")
    ax.set_ylabel("Window width (mV)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axvspan(5, 10, alpha=0.1, color=COLOR_GREEN, label="Recommended range")
    ax.text(7.5, max(widths_ss)*0.85, "Recommended\n\u03b2 range",
            fontsize=8, ha="center", color=COLOR_GREEN)

    fig.tight_layout()
    return fig


def make_ch6_custom_vjp_page():
    """Page explaining custom_vjp mechanism."""
    fig, ax, y = new_text_page("6. Surrogate Gradients (continued)",
                                "The custom_vjp implementation in JAX")

    y = add_text(ax, y, "6.4  Implementation via custom_vjp", fontsize=12, weight="bold")
    y = add_text(ax, y, dedent("""\
    JAX's custom_vjp decorator allows defining separate forward and backward computations \
    for any function. For the spike detection:"""))
    y -= 0.005

    y = add_text(ax, y, "The implementation defines three functions:", indent=0.02)
    y -= 0.005
    y = add_bullet(ax, y, "heaviside_surrogate(x): The primal function \u2014 returns hard Heaviside "
                   "H(x). This is what executes during the forward pass, so the simulation "
                   "produces real binary spikes.")
    y = add_bullet(ax, y, "heaviside_fwd(x): Returns (output, residuals). The residuals (here just x) "
                   "are saved for use in the backward pass.")
    y = add_bullet(ax, y, "heaviside_bwd(residuals, g): Receives the saved residuals and the upstream "
                   "gradient g. Returns the local gradient computed using the surrogate derivative "
                   "instead of the true derivative.")
    y -= 0.005

    y = add_text(ax, y, "The backward pass computes:", indent=0.02)
    y = add_equation(ax, y, r"$\text{output gradient} = g \cdot \tilde{\sigma}'(x)$")
    y = add_text(ax, y, dedent("""\
    where g is the incoming gradient from downstream operations (the chain rule factor \
    from the loss through the reset mechanism) and \u03c3\u0303'(x) is the chosen surrogate \
    derivative evaluated at the saved residual x = v \u2212 v_threshold."""))
    y -= 0.005

    y = add_text(ax, y, "6.5  How Gradients Flow Through the Continuous Reset", fontsize=12, weight="bold")
    y = add_text(ax, y, dedent("""\
    The continuous reset formulation v_new = s\u00b7v_reset + (1\u2212s)\u00b7v\u0303 has derivatives via \
    the product rule:"""))
    y = add_equation(ax, y,
        r"$\frac{\partial v_\mathrm{new}}{\partial s} = v_\mathrm{reset} - \tilde{v}$",
        fontsize=12)
    y = add_equation(ax, y,
        r"$\frac{\partial v_\mathrm{new}}{\partial \tilde{v}} = 1 - s$",
        fontsize=12)
    y = add_equation(ax, y,
        r"$\frac{\partial v_\mathrm{new}}{\partial v_\mathrm{reset}} = s$",
        fontsize=12)

    y = add_text(ax, y, dedent("""\
    During a spike (s \u2248 1): \u2202v_new/\u2202v\u0303 \u2248 0 (pre-reset voltage is 'forgotten') and \
    \u2202v_new/\u2202v_reset \u2248 1 (gradient flows fully to v_reset). Between spikes (s \u2248 0): \
    \u2202v_new/\u2202v\u0303 \u2248 1 (gradient flows through dynamics) and \u2202v_new/\u2202v_reset \u2248 0."""))
    y -= 0.005

    y = add_text(ax, y, dedent("""\
    The gradient through s is where the surrogate enters:"""))
    y = add_equation(ax, y,
        r"$\frac{\partial v_\mathrm{new}}{\partial v} = \frac{\partial v_\mathrm{new}}{\partial s} \cdot \frac{\partial s}{\partial v} + \frac{\partial v_\mathrm{new}}{\partial \hat{v}} \cdot \frac{\partial \hat{v}}{\partial v}$",
        fontsize=11)

    y = add_text(ax, y, dedent("""\
    The first term provides gradient signal about spike timing (mediated by the surrogate). \
    The second term carries the standard dynamics gradient. Together, they enable end-to-end \
    differentiation through the spiking neuron."""))

    return fig


def make_ch7_page1():
    """Chapter 7: Loss Function Design."""
    fig, ax, y = new_text_page("7. Loss Function Design for Spiking Models",
                                "How the choice of loss shapes the gradient landscape")

    y = add_text(ax, y, dedent("""\
    The loss function transforms a voltage trace (or spike train) into a scalar that the \
    optimizer can minimize. The choice of loss fundamentally determines what the optimizer \
    'sees' \u2014 it defines the gradient landscape that guides parameter updates. Different \
    losses create radically different optimization problems (Bellec et al., 2020)."""))
    y -= 0.005

    y = add_text(ax, y, "7.1  Mean Squared Error (MSE) on Voltage", fontsize=12, weight="bold")
    y = add_equation(ax, y, r"$L_\mathrm{MSE} = \frac{1}{T}\sum_{t=1}^{T} (v_t^\mathrm{sim} - v_t^\mathrm{exp})^2$")
    y = add_text(ax, y, dedent("""\
    Gradient: \u2202L/\u2202v_t = (2/T)(v_t \u2212 v_t\u1d49\u02e3\u1d56). Each timestep contributes independently, \
    proportional to the local voltage error."""))
    y = add_text(ax, y, "Failure mode:", weight="bold", indent=0.02)
    y = add_text(ax, y, dedent("""\
    A spike misaligned by just 1ms creates ~100 mV error at two timesteps (spike vs. \
    subthreshold). The optimizer discovers that eliminating spikes entirely reduces MSE \
    more than improving their timing. This drives convergence to non-spiking solutions \
    \u2014 a degenerate local minimum."""), indent=0.04)
    y -= 0.005

    y = add_text(ax, y, "7.2  Van Rossum Distance (Spike Train Metric)", fontsize=12, weight="bold")
    y = add_equation(ax, y, r"$L_\mathrm{VR} = \frac{\Delta t}{\tau} \sum_t \left[\left(K_\tau * s^\mathrm{sim}\right)_t - \left(K_\tau * s^\mathrm{exp}\right)_t\right]^2$")
    y = add_text(ax, y, dedent("""\
    where K_\u03c4(t) = exp(\u2212t/\u03c4) is an exponential kernel and * denotes convolution \
    (van Rossum, 2001). This convolves each spike train with a decaying exponential, \
    then computes the L\u00b2 distance between the filtered signals."""))
    y = add_text(ax, y, "Key property:", weight="bold", indent=0.02)
    y = add_text(ax, y, dedent("""\
    The convolution is naturally differentiable via JAX. The gradient of the Van Rossum \
    distance w.r.t. each soft spike indicator s_t propagates through the convolution kernel, \
    providing temporal credit assignment: a misplaced spike creates a smooth error signal \
    over a window of width ~\u03c4, not a point error. The \u03c4 parameter controls tolerance \u2014 \
    large \u03c4 for coarse matching, small \u03c4 for precise timing."""), indent=0.04)
    y -= 0.005

    y = add_text(ax, y, "7.3  Soft-DTW (Dynamic Time Warping)", fontsize=12, weight="bold")
    y = add_text(ax, y, dedent("""\
    Soft-DTW (Cuturi & Blondel, 2017) replaces the hard min in DTW with a differentiable \
    soft-minimum, allowing temporal alignment to be learned via gradient descent. It builds \
    a T\u00d7T cost matrix and finds the optimal warping path. However, when applied to raw \
    voltage traces, subthreshold dynamics dominate the cost (they occupy ~90% of the trace), \
    diluting the spike alignment signal."""))
    y -= 0.005

    y = add_text(ax, y, "7.4  Deistler Summary Statistics", fontsize=12, weight="bold")
    y = add_text(ax, y, dedent("""\
    Deistler et al. (2025) propose matching summary statistics (mean and standard deviation \
    of voltage in pre-stimulus and stimulus windows). This is fully differentiable and \
    provides smooth gradients, but loses all temporal structure \u2014 two traces with opposite \
    spike patterns but similar statistics would have near-zero loss."""))

    return fig


def make_loss_landscape_figure():
    """Figure: Conceptual loss landscapes for different loss functions."""
    fig, axes = plt.subplots(2, 2, figsize=(A4_W, 6.5))
    fig.suptitle("Figure 7.1: Loss Landscape Characteristics by Loss Function",
                 fontsize=11, fontweight="bold", y=1.0)

    np.random.seed(42)

    # (a) MSE - sharp cliffs at spike boundaries
    ax = axes[0, 0]
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    # MSE-like: smooth except at spike boundaries
    Z_mse = X**2 + Y**2 + 3*np.exp(-((X-0.5)**2 + (Y+0.3)**2)/0.05)  # sharp cliff
    Z_mse += 2*np.exp(-((X+0.7)**2 + (Y-0.5)**2)/0.03)
    # Add a degenerate minimum (non-spiking)
    Z_mse -= 1.5*np.exp(-((X+1.2)**2 + (Y-1.3)**2)/0.3)

    ax.contourf(X, Y, Z_mse, levels=20, cmap="RdYlBu_r", alpha=0.8)
    ax.contour(X, Y, Z_mse, levels=10, colors="black", linewidths=0.3, alpha=0.5)
    ax.plot(-1.2, 1.3, "kx", markersize=10, markeredgewidth=2)
    ax.annotate("Non-spiking\nminimum", xy=(-1.2, 1.3), xytext=(-0.3, 1.6),
               fontsize=7, arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_title("(a) MSE on voltage", fontsize=10)
    ax.set_xlabel("Parameter 1")
    ax.set_ylabel("Parameter 2")

    # (b) Van Rossum - smoother with temporal tolerance
    ax = axes[0, 1]
    Z_vr = 0.8*X**2 + 0.6*Y**2 + 0.3*np.sin(3*X)*np.cos(2*Y)
    Z_vr += 0.5*np.exp(-((X-0.8)**2 + Y**2)/0.5)
    ax.contourf(X, Y, Z_vr, levels=20, cmap="RdYlBu_r", alpha=0.8)
    ax.contour(X, Y, Z_vr, levels=10, colors="black", linewidths=0.3, alpha=0.5)
    ax.plot(0, 0, "k*", markersize=10)
    ax.annotate("Global min\n(correct params)", xy=(0, 0), xytext=(0.8, -1.2),
               fontsize=7, arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_title("(b) Van Rossum distance", fontsize=10)
    ax.set_xlabel("Parameter 1")

    # (c) Soft-DTW - smooth but broad basin
    ax = axes[1, 0]
    Z_dtw = 0.5*(X**2 + Y**2) + 0.1*np.sin(5*X) + 0.1*np.cos(5*Y)
    # Add subthreshold domination (flat plateau)
    Z_dtw += 0.3 * np.clip(1 - (X**2 + Y**2), 0, 1)
    ax.contourf(X, Y, Z_dtw, levels=20, cmap="RdYlBu_r", alpha=0.8)
    ax.contour(X, Y, Z_dtw, levels=10, colors="black", linewidths=0.3, alpha=0.5)
    ax.set_title("(c) Soft-DTW on voltage", fontsize=10)
    ax.set_xlabel("Parameter 1")
    ax.set_ylabel("Parameter 2")
    ax.text(-1.5, -1.8, "Subthreshold\ndominates", fontsize=7, color="white",
            fontweight="bold")

    # (d) Deistler - very smooth but many equivalent minima
    ax = axes[1, 1]
    Z_dei = 0.3*(X**2 + Y**2) + 0.05*X*Y
    # Ring of near-equivalent minima (many params give same stats)
    r = np.sqrt(X**2 + Y**2)
    Z_dei += 0.5 * np.abs(r - 1) * np.exp(-(r-1)**2/0.3)
    ax.contourf(X, Y, Z_dei, levels=20, cmap="RdYlBu_r", alpha=0.8)
    ax.contour(X, Y, Z_dei, levels=10, colors="black", linewidths=0.3, alpha=0.5)
    theta_ring = np.linspace(0, 2*np.pi, 100)
    # Show ring of degenerate solutions
    ax.set_title("(d) Deistler summary stats", fontsize=10)
    ax.set_xlabel("Parameter 1")
    ax.text(0.5, -1.8, "Smooth but\nlosses temporal info", fontsize=7, color="white",
            fontweight="bold")

    fig.tight_layout()
    return fig


def make_loss_gradient_figure():
    """Figure: How each loss creates different gradient signals on a voltage trace."""
    fig, axes = plt.subplots(4, 1, figsize=(A4_W, 7), sharex=True)
    fig.suptitle("Figure 7.2: Gradient Signal From Different Loss Functions",
                 fontsize=11, fontweight="bold", y=1.0)

    np.random.seed(42)
    T = 500
    dt = 0.1  # ms
    t = np.arange(T) * dt

    # Create synthetic voltage traces
    v_exp = -65 * np.ones(T)
    v_sim = -65 * np.ones(T)

    # Experimental spikes at t=15, 30, 42 ms
    spike_times_exp = [150, 300, 420]
    spike_times_sim = [155, 295, 430]  # slightly shifted

    for st in spike_times_exp:
        v_exp[st:st+3] = [-40, 20, -55]
    for st in spike_times_sim:
        v_sim[st:st+3] = [-40, 20, -55]

    # Add some subthreshold fluctuation
    v_exp += 3*np.sin(2*np.pi*t/20) + np.random.randn(T)*0.5
    v_sim += 2.5*np.sin(2*np.pi*t/20 + 0.2) + np.random.randn(T)*0.5

    # (a) Voltage traces
    ax = axes[0]
    ax.plot(t, v_exp, color=COLOR_BLUE, lw=1, label="Target", alpha=0.8)
    ax.plot(t, v_sim, color=COLOR_ACCENT, lw=1, label="Simulated", alpha=0.8)
    ax.set_ylabel("Voltage (mV)")
    ax.set_title("(a) Voltage traces", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-75, 30)

    # (b) MSE gradient
    ax = axes[1]
    mse_grad = 2/T * (v_sim - v_exp)
    ax.fill_between(t, mse_grad, alpha=0.4, color=COLOR_ORANGE)
    ax.plot(t, mse_grad, color=COLOR_ORANGE, lw=0.8)
    ax.set_ylabel(r"$\partial L / \partial v_t$")
    ax.set_title("(b) MSE gradient: huge spikes dominate", fontsize=10, loc="left")
    # Highlight spike errors
    for st in spike_times_exp + spike_times_sim:
        if abs(mse_grad[min(st+1, T-1)]) > 0.1:
            ax.axvline(st*dt, color="red", ls=":", lw=0.5, alpha=0.5)

    # (c) Van Rossum gradient (conceptual)
    ax = axes[2]
    # Approximate: exponential kernel convolution difference
    tau = 10  # ms
    kernel_len = int(5*tau/dt)
    kernel = np.exp(-np.arange(kernel_len) * dt / tau)
    kernel /= kernel.sum()

    # Create soft spike trains
    s_exp = np.zeros(T)
    s_sim = np.zeros(T)
    for st in spike_times_exp:
        s_exp[st] = 1.0
    for st in spike_times_sim:
        s_sim[st] = 1.0

    f_exp = np.convolve(s_exp, kernel, mode="same")
    f_sim = np.convolve(s_sim, kernel, mode="same")
    vr_grad = 2 * (f_sim - f_exp)

    ax.fill_between(t, vr_grad, alpha=0.4, color=COLOR_GREEN)
    ax.plot(t, vr_grad, color=COLOR_GREEN, lw=0.8)
    ax.set_ylabel(r"$\partial L / \partial s_t$")
    ax.set_title("(c) Van Rossum gradient: smooth temporal credit", fontsize=10, loc="left")

    # (d) Deistler gradient (constant)
    ax = axes[3]
    dei_grad = np.ones(T) * 0.002  # Uniform contribution to mean/std
    # Slightly higher where voltage deviates more from mean
    mean_diff = np.sign(v_sim - np.mean(v_sim)) * 0.001
    dei_grad += mean_diff
    ax.fill_between(t, dei_grad, alpha=0.4, color=COLOR_BLUE)
    ax.plot(t, dei_grad, color=COLOR_BLUE, lw=0.8)
    ax.set_ylabel(r"$\partial L / \partial v_t$")
    ax.set_title("(d) Deistler gradient: uniform, no temporal structure", fontsize=10, loc="left")
    ax.set_xlabel("Time (ms)")

    fig.tight_layout()
    return fig


def make_ch8_page1():
    """Chapter 8: Putting It All Together."""
    pw = PageWriter("8. Putting It All Together",
                                "The complete gradient pipeline for AdEx optimization")

    pw.text(dedent("""\
    We now have all the pieces to understand the full gradient computation pipeline. \
    Let us trace how a single training step transforms parameter values into gradient \
    updates, end to end."""))
    pw.gap()

    pw.text("8.1  The Optimization Objective", fontsize=12, weight="bold")
    pw.text(dedent("""\
    Given experimental voltage trace v_exp recorded from a biological neuron in response \
    to current injection I_ext, find parameters \u03b8* that minimize:"""))
    pw.equation(r"$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \; \mathcal{L}\!\left(\mathrm{sim}(\boldsymbol{\theta}, I_\mathrm{ext}),\; v_\mathrm{exp}\right)$")
    pw.text(dedent("""\
    where sim(\u03b8, I_ext) runs the AdEx integrator and returns the simulated voltage trace. \
    The function composition is: \u03b8 \u2192 simulate \u2192 voltage trace \u2192 loss \u2192 scalar."""))
    pw.gap()

    pw.text("8.2  Step-by-Step: One Training Iteration", fontsize=12, weight="bold")
    pw.gap()

    pw.text("Step 1: Parameter preparation", weight="bold", indent=0.02)
    pw.text(dedent("""\
    If using sigmoid reparameterization, transform unconstrained parameters to the \
    constrained domain: \u03b8 = lower + (upper \u2212 lower) \u00b7 \u03c3(\u03b8_unconstrained). This ensures \
    parameters stay within physically meaningful bounds while gradients flow smoothly."""),
                 indent=0.04)
    pw.gap()

    pw.text("Step 2: Forward simulation (5000 Euler steps)", weight="bold", indent=0.02)
    pw.text(dedent("""\
    At each timestep, compute the AdEx dynamics using the current parameters. The \
    surrogate_fn returns binary spikes in the forward pass but stores residuals for the \
    backward pass. The continuous reset blends v_reset and v based on the spike indicator. \
    Output: voltage[T] and spikes[T]."""), indent=0.04)
    pw.gap()

    pw.text("Step 3: Loss evaluation", weight="bold", indent=0.02)
    pw.text(dedent("""\
    The loss function reduces the T-dimensional voltage trace to a single scalar. This \
    scalar is the starting point for the backward pass. The choice of loss function \
    determines the initial gradient signal \u2202L/\u2202v_t at each timestep."""), indent=0.04)
    pw.gap()

    pw.text("Step 4: Backward pass (automatic via JAX)", weight="bold", indent=0.02)
    pw.text(dedent("""\
    JAX walks backward through the computational graph. At spike events, custom_vjp \
    substitutes the surrogate derivative. At each timestep, the chain rule accumulates \
    \u2202L/\u2202\u03b8 contributions. The gradient decays through leak dynamics (~0.9997\u1d40 per step \
    backward). The result: one gradient value per trainable parameter."""), indent=0.04)
    pw.gap()

    pw.text("Step 5: Parameter update", weight="bold", indent=0.02)
    pw.text(dedent("""\
    The Adam optimizer adjusts each parameter using adaptive learning rates based on \
    gradient history. Optional gradient clipping prevents catastrophically large updates. \
    If not using sigmoid reparameterization, hard clipping enforces parameter bounds \
    (but kills gradients at boundaries)."""), indent=0.04)
    pw.gap()

    pw.text("8.3  Why Each Component Matters", fontsize=12, weight="bold")
    pw.text(dedent("""\
    The gradient \u2202L/\u2202\u03b8 is a product of many factors along the chain. If any factor is \
    zero or near-zero, the entire gradient vanishes:"""))
    pw.bullet("Loss \u2192 voltage: loss function must provide informative, non-misleading signal")
    pw.bullet("Voltage \u2192 spike: surrogate must have sufficient width (\u03b2 = 5\u201310)")
    pw.bullet("Spike \u2192 reset: continuous formulation must be used (not hard conditional)")
    pw.bullet("Through time: leak decay limits useful gradient propagation depth")
    pw.bullet("Parameters \u2192 update: bounds must not clip gradients (use sigmoid reparam)")

    return pw.figures


def make_pipeline_figure():
    """Figure: Complete gradient pipeline diagram."""
    fig, ax = plt.subplots(figsize=(A4_W, 5.5))
    fig.suptitle("Figure 8.1: Complete Gradient Pipeline for AdEx Parameter Optimization",
                 fontsize=11, fontweight="bold", y=0.98)
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-3.5, 3.5)
    ax.axis("off")

    # Forward pass (top row)
    fw_boxes = [
        (0.5, 2, "$\\boldsymbol{\\theta}$", "Parameters", COLOR_BLUE),
        (2.5, 2, "$\\sigma(\\cdot)$", "Reparam", COLOR_GREEN),
        (4.5, 2, "Simulate", "5000 steps", COLOR_BLUE),
        (6.8, 2, "$v(t), s(t)$", "Traces", COLOR_BLUE),
        (9, 2, "$L$", "Loss", COLOR_ACCENT),
    ]

    for x, yy, label, sublabel, color in fw_boxes:
        w, h = 1.3, 0.7
        if label in ["$L$", "$\\boldsymbol{\\theta}$"]:
            w = 0.8
        rect = FancyBboxPatch((x - w/2, yy - h/2), w, h,
                              boxstyle="round,pad=0.08",
                              facecolor=COLOR_LIGHT, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, yy + 0.05, label, ha="center", va="center", fontsize=11, color=color)
        ax.text(x, yy - 0.25, sublabel, ha="center", va="center", fontsize=7, color="gray")

    # Forward arrows
    for i in range(len(fw_boxes)-1):
        x1 = fw_boxes[i][0] + (0.4 if fw_boxes[i][2] in ["$L$", "$\\boldsymbol{\\theta}$"] else 0.65)
        x2 = fw_boxes[i+1][0] - (0.4 if fw_boxes[i+1][2] in ["$L$", "$\\boldsymbol{\\theta}$"] else 0.65)
        ax.annotate("", xy=(x2, 2), xytext=(x1, 2),
                   arrowprops=dict(arrowstyle="-|>", color=COLOR_BLUE, lw=1.5))

    ax.text(5, 3.2, "FORWARD PASS", fontsize=12, fontweight="bold",
            color=COLOR_BLUE, ha="center")

    # Backward pass (bottom row)
    bw_labels = [
        (9, -0.3, r"$\frac{\partial L}{\partial L} = 1$"),
        (6.8, -0.3, r"$\frac{\partial L}{\partial v_t}$"),
        (4.5, -0.3, r"$\frac{\partial L}{\partial \mathbf{s}_t}$"),
        (2.5, -0.3, r"$\frac{\partial L}{\partial \boldsymbol{\theta}}$"),
        (0.5, -0.3, "Update $\\boldsymbol{\\theta}$"),
    ]

    for x, yy, label in bw_labels:
        ax.text(x, yy, label, ha="center", va="center", fontsize=9, color=COLOR_ACCENT,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#fce4ec",
                         edgecolor=COLOR_ACCENT, alpha=0.8))

    # Backward arrows
    for i in range(len(bw_labels)-1):
        x1 = bw_labels[i][0] - 0.5
        x2 = bw_labels[i+1][0] + 0.6
        ax.annotate("", xy=(x2, -0.3), xytext=(x1, -0.3),
                   arrowprops=dict(arrowstyle="-|>", color=COLOR_ACCENT, lw=1.5))

    ax.text(5, -1.1, "BACKWARD PASS (reverse-mode AD)", fontsize=12, fontweight="bold",
            color=COLOR_ACCENT, ha="center")

    # Annotations for key challenges
    challenges = [
        (4.5, -2.0, "Surrogate gradient\nreplaces H'(x) here", 4.5, -0.8),
        (6.8, -2.0, "Loss choice determines\ninitial gradient signal", 6.8, -0.8),
        (2.5, -2.5, "Gradient decays \u00d70.9997\nper timestep backward", 3.5, -0.8),
        (8, -2.5, "Sigmoid reparam\navoids bound clipping", 1.5, -0.8),
    ]

    for tx, ty, text, px, py in challenges:
        ax.text(tx, ty, text, ha="center", va="center", fontsize=7,
                color=COLOR_TEAL, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#e8eaf6",
                         edgecolor=COLOR_TEAL, alpha=0.6))
        ax.annotate("", xy=(px, py), xytext=(tx, ty + 0.25),
                   arrowprops=dict(arrowstyle="->", color=COLOR_TEAL, lw=0.8,
                                  connectionstyle="arc3,rad=0.2", alpha=0.5))

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def make_references_page():
    """References page."""
    fig, ax, y = new_text_page("References")

    refs = [
        "Baydin, A.G., Pearlmutter, B.A., Radul, A.A. & Siskind, J.M. (2018). Automatic differentiation in machine learning: A survey. JMLR, 18(153), 1-43.",
        "Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R. & Maass, W. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. Nature Communications, 11, 3625.",
        "Bengio, Y., Simard, P. & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. IEEE TNN, 5(2), 157-166.",
        "Bengio, Y., L\u00e9onard, N. & Courville, A. (2013). Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv:1308.3432.",
        "Bradbury, J. et al. (2018). JAX: Composable transformations of Python+NumPy programs. github.com/jax-ml/jax.",
        "Brette, R. & Gerstner, W. (2005). Adaptive exponential integrate-and-fire model as an effective description of neuronal activity. J. Neurophysiol., 94(5), 3637-3642.",
        "Cuturi, M. & Blondel, M. (2017). Soft-DTW: A differentiable loss function for time series. ICML, 894-903.",
        "Deistler, M. et al. (2025). Differentiable simulation enables large-scale training of detailed biophysical models of neural dynamics. bioRxiv.",
        "Goodfellow, I., Bengio, Y. & Courville, A. (2016). Deep Learning. MIT Press. Ch. 6.5, 10.2.",
        "Griewank, A. & Walther, A. (2008). Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation. 2nd ed. SIAM.",
        "Gygax, J. & Zenke, F. (2025). Elucidating the theoretical underpinnings of surrogate gradient learning in spiking neural networks. Nature Communications, 16, 790.",
        "Hochreiter, S. (1991). Untersuchungen zu dynamischen neuronalen Netzen. Diploma thesis, TU Munich.",
        "Kingma, D.P. & Ba, J. (2015). Adam: A method for stochastic optimization. ICLR.",
        "Klos, C. et al. (2024). Pseudospikes: A unified framework for surrogate gradient calibration. In preparation.",
        "Neftci, E.O., Mostafa, H. & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks. IEEE Signal Processing Magazine, 36(6), 51-63.",
        "Nocedal, J. & Wright, S.J. (2006). Numerical Optimization. 2nd ed. Springer.",
        "Rudin, W. (1976). Principles of Mathematical Analysis. 3rd ed. McGraw-Hill.",
        "Rumelhart, D.E., Hinton, G.E. & Williams, R.J. (1986). Learning representations by back-propagating errors. Nature, 323, 533-536.",
        "van Rossum, M.C.W. (2001). A novel spike distance. Neural Computation, 13(4), 751-763.",
        "Werbos, P.J. (1990). Backpropagation through time: What it does and how to do it. Proc. IEEE, 78(10), 1550-1560.",
        "Zenke, F. & Ganguli, S. (2018). SuperSpike: Supervised learning in multilayer spiking neural networks. Neural Computation, 30(6), 1514-1541.",
    ]

    for ref in refs:
        y = add_text(ax, y, ref, fontsize=8, indent=0.03, line_spacing=0.017)
        y -= 0.004
        if y < 0.03:
            break

    return fig


# ===========================================================================
# MAIN: Assemble the PDF
# ===========================================================================

def main():
    output_path = "/Users/paulmayer/Projects/university/40_thesis/2_thesis/backprop_study_guide.pdf"

    pages = [
        ("Title Page", make_title_page),
        ("Ch1: Prerequisites", make_ch1_page1),
        ("Fig 1.1: Chain Rule", make_ch1_chain_rule_figure),
        ("Ch2: Computational Graphs", make_ch2_page1),
        ("Ch3: Backpropagation", make_ch3_page1),
        ("Fig 3.1: Forward/Backward", make_ch3_backprop_figure),
        ("Ch4: BPTT", make_ch4_page1),
        ("Fig 4.1: Unrolled Graph", make_ch4_bptt_figure),
        ("Fig 4.2: Vanishing Gradients", make_vanishing_gradient_figure),
        ("Ch4b: Why Differentiable", make_ch4b_why_differentiable),
        ("Fig 4b.1: Differentiability", make_ch4b_differentiability_figure),
        ("Ch4c: BPTT Detailed", make_ch4c_bptt_detailed),
        ("Fig 4c.1: Gradient Paths", make_ch4c_gradient_path_figure),
        ("Ch4d: Spike-Gated Grads", make_ch4d_spike_gated_gradients),
        ("Fig 4d.1: Gradient Map", make_ch4d_gradient_existence_figure),
        ("Ch4e: Loss to Params", make_ch4e_loss_to_param),
        ("Ch5: Spike Problem", make_ch5_page1),
        ("Fig 5.1: Heaviside", make_heaviside_figure),
        ("Ch6: Surrogates", make_ch6_page1),
        ("Fig 6.1: Surrogate Comparison", make_surrogate_comparison_figure),
        ("Ch6 cont: custom_vjp", make_ch6_custom_vjp_page),
        ("Ch7: Loss Functions", make_ch7_page1),
        ("Fig 7.1: Loss Landscapes", make_loss_landscape_figure),
        ("Fig 7.2: Loss Gradients", make_loss_gradient_figure),
        ("Ch8: Full Pipeline", make_ch8_page1),
        ("Fig 8.1: Pipeline Diagram", make_pipeline_figure),
        ("References", make_references_page),
    ]

    total_pages = 0
    with PdfPages(output_path) as pdf:
        for name, generator in pages:
            print(f"  Generating: {name}")
            result = generator()
            # PageWriter-based generators return a list of figures
            figs = result if isinstance(result, list) else [result]
            for i, fig in enumerate(figs):
                pdf.savefig(fig, bbox_inches="tight", pad_inches=0.3)
                plt.close(fig)
                total_pages += 1
            if len(figs) > 1:
                print(f"    -> {len(figs)} pages (auto-paginated)")

    print(f"\nPDF saved to: {output_path}")
    print(f"Total pages: {total_pages}")


if __name__ == "__main__":
    main()
