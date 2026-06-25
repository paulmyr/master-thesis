"""Diagnose why gradient descent underperforms Nelder-Mead on the subthreshold
benchmark, and find a gradient config that beats NM at the same budget.

The subthreshold AdEx fit (C_m, E_L, g_L vs an MSE voltage trace) is a smooth,
mildly-nonlinear least-squares problem — gradient descent *should* converge in a
few steps. This script pins the misconfiguration and retunes:

  1. Gradient-scale diagnostic: per-parameter init |dL/dtheta| in raw vs logit
     (sigmoid-transform) space, plus "Adam steps to cross 50% of each param's
     range at the current lr" — shows a single global lr can't move all three
     raw params (ranges differ ~30x), motivating the transform.
  2. Optimizer x transform x lr sweep at the 25-eval budget vs paired NM.
  3. Budget/convergence curve for the best gradient config vs NM.

Reuses `benchmark_subthreshold` for scenarios, cell/loss build, NM, evaluation.
Run with `python subthreshold_grad_tuning.py`. Figures -> figures/.
"""

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

import benchmark_subthreshold as B
from ADoptEX.core.parameters import PARAM_BOUNDS
from ADoptEX.training.trainer import (TrainingConfig, build_param_transform,
                                      nudge_from_bounds, train)

# --- Tuning config ---------------------------------------------------------

SUBSET_DIRS = 4                       # directions/delta for the tuning subset
OPTIMIZERS = ["adam", "sgd", "polyak"]
TRANSFORMS = [True, False]
LRS = [0.05, 0.1, 0.2, 0.5, 1.0]
BUDGETS = [5, 10, 15, 25, 50]
PARAMS = B.TRAINABLE                   # ["C_m", "E_L", "g_L"]

FIGS = Path(__file__).parent / "figures"
FIGS.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times"],
    "mathtext.fontset": "cm", "axes.labelsize": 10, "axes.titlesize": 10,
    "axes.titleweight": "regular", "axes.linewidth": 0.6, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.linewidth": 0.4, "grid.alpha": 0.35, "pdf.fonttype": 42,
    "ps.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
})
COLORS = {"grad": "#0072B2", "nm": "#E69F00"}


# --- Helpers ---------------------------------------------------------------

def _pname(k):
    k = k.replace("AdEx_", "")
    return "C_m" if k == "capacitance" else k


def per_param_abs_grad(grad_tree):
    out = {}
    for d in grad_tree:
        for k, v in d.items():
            out[_pname(k)] = float(np.abs(np.asarray(v).ravel()[0]))
    return out


def init_grads(trace):
    """Per-parameter |dL/dtheta| at the fixed init, in raw and logit space."""
    _, handles, loss_fn = B.build(trace)
    raw = per_param_abs_grad(jax.grad(loss_fn)(handles))

    tf = build_param_transform(handles, PARAM_BOUNDS)
    h_logit = tf.inverse(nudge_from_bounds(handles, PARAM_BOUNDS))
    loss_logit = lambda p: loss_fn(tf.forward(p))
    logit = per_param_abs_grad(jax.grad(loss_logit)(h_logit))
    return raw, logit


def run_grad_cfg(trace, opt, lr, transform, budget):
    """Train one gradient config; return (clean_final_loss, iters_to_tol)."""
    _, handles, loss_fn = B.build(trace)
    cfg = TrainingConfig(
        optimizer=opt, learning_rate=lr, n_epochs=budget,
        surrogate_type=B.GRAD["surrogate_type"],
        surrogate_slope=B.GRAD["surrogate_slope"],
        use_param_transform=transform, clip_to_bounds=(not transform),
        return_best=True, verbose=False)
    with B._silence():
        result = train(loss_fn, handles, cfg, initial_params=B.INIT)
    fitted = result.get_params_dict()
    curve = [float(x) for x in result.loss_history]
    return B.clean_mse(fitted, trace), B.iters_to_tol(curve)


def run_nm_budget(trace, maxfev):
    """NM with a controllable eval budget; return (clean_final_loss, iters_to_tol)."""
    cell, _, loss_fn = B.build(trace)
    loss_jit = jax.jit(loss_fn)
    keys = [list(p.keys())[0] for p in cell.get_parameters()]
    curve = []

    def eval_loss(x):
        p = [{k: jnp.array([float(x[i])])} for i, k in enumerate(keys)]
        try:
            v = float(loss_jit(p))
            v = v if math.isfinite(v) else 1e6
        except Exception:
            v = 1e6
        curve.append(v)
        return v

    x0 = np.array([B.INIT[n] for n in PARAMS])
    bounds = [(PARAM_BOUNDS[n].min, PARAM_BOUNDS[n].max) for n in PARAMS]
    res = scipy.optimize.minimize(
        eval_loss, x0=x0, method="Nelder-Mead", bounds=bounds,
        options={"maxfev": maxfev, "adaptive": True, "xatol": 1e-3, "fatol": 1e-5})
    fitted = {n: float(res.x[i]) for i, n in enumerate(PARAMS)}
    return B.clean_mse(fitted, trace), B.iters_to_tol(curve)


def _med(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


# --- Main ------------------------------------------------------------------

def main():
    B.N_DIRS = SUBSET_DIRS                       # smaller, representative subset
    scenarios = B.generate_scenarios()
    traces = [s["trace"] for s in scenarios]
    print(f"Tuning subset: {len(traces)} scenarios "
          f"(bases={B.BASES}, deltas={B.DELTAS}, {SUBSET_DIRS} dirs each)\n")

    # 1) Gradient-scale diagnostic ------------------------------------------
    raws, logits = [], []
    for tr in traces:
        r, l = init_grads(tr)
        raws.append(r); logits.append(l)
    raw_med = {p: _med([r[p] for r in raws]) for p in PARAMS}
    logit_med = {p: _med([l[p] for l in logits]) for p in PARAMS}
    rng = {p: PARAM_BOUNDS[p].max - PARAM_BOUNDS[p].min for p in PARAMS}
    lr_ref = B.GRAD["lr"]
    steps_50 = {p: 0.5 * rng[p] / lr_ref for p in PARAMS}  # Adam ~lr/step in raw units

    print("=" * 70)
    print(f"GRADIENT-SCALE DIAGNOSTIC (median over subset, current lr={lr_ref})")
    print("=" * 70)
    print(f"{'param':<6}{'range':>8}{'|grad| raw':>14}{'|grad| logit':>14}"
          f"{'Adam steps/50% range':>22}")
    for p in PARAMS:
        print(f"{p:<6}{rng[p]:>8.1f}{raw_med[p]:>14.3g}{logit_med[p]:>14.3g}"
              f"{steps_50[p]:>22.0f}")
    print("\nRaw |grad| spans orders of magnitude across params and 'steps/50% "
          "range'\nexceeds the 25-eval budget for C_m/E_L -> a single global lr in raw "
          "units\ncannot move all three. The logit (sigmoid-transform) space equalizes "
          "scale.\n")

    # 2) Optimizer x transform x lr sweep at the 25-eval budget --------------
    nm_final = [run_nm_budget(tr, B.MAX_ITERS)[0] for tr in traces]
    nm_med = _med(nm_final)
    print("=" * 70)
    print(f"SWEEP @ budget={B.MAX_ITERS}  (NM median final MSE = {nm_med:.4g})")
    print("=" * 70)
    print(f"{'optimizer':<10}{'transform':<10}{'lr':>6}{'med final':>12}"
          f"{'P(conv)':>9}{'med iters':>11}{'win% vs NM':>12}")
    grid = []
    for opt in OPTIMIZERS:
        for tf in TRANSFORMS:
            for lr in LRS:
                finals, iters = [], []
                for tr in traces:
                    fl, it = run_grad_cfg(tr, opt, lr, tf, B.MAX_ITERS)
                    finals.append(fl); iters.append(it)
                p_conv = float(np.mean([f <= B.CONV_TOL for f in finals]))
                med_it = _med([it for it in iters if it >= 0])
                win = float(np.mean([g < n for g, n in zip(finals, nm_final)]))
                row = dict(opt=opt, tf=tf, lr=lr, med_final=_med(finals),
                           p_conv=p_conv, med_iters=med_it, win=win)
                grid.append(row)
                print(f"{opt:<10}{str(tf):<10}{lr:>6}{row['med_final']:>12.4g}"
                      f"{p_conv:>9.2f}{med_it:>11.1f}{100*win:>11.0f}%")

    grid.sort(key=lambda r: (r["med_final"], -r["win"]))
    best = grid[0]
    print(f"\nBEST: optimizer={best['opt']} transform={best['tf']} lr={best['lr']} "
          f"-> med final={best['med_final']:.4g}, P(conv)={best['p_conv']:.2f}, "
          f"med iters={best['med_iters']:.1f}, win vs NM={100*best['win']:.0f}%")

    # 3) Budget / convergence curve: best grad config vs NM -----------------
    print("\n" + "=" * 70)
    print("BUDGET CURVE (median final MSE; P(conv))")
    print("=" * 70)
    print(f"{'budget':>8}{'grad med':>12}{'grad Pconv':>12}{'nm med':>12}{'nm Pconv':>12}")
    g_curve, n_curve = [], []
    for bud in BUDGETS:
        gf = [run_grad_cfg(tr, best["opt"], best["lr"], best["tf"], bud)[0]
              for tr in traces]
        nf = [run_nm_budget(tr, max(bud, 4))[0] for tr in traces]
        gp = float(np.mean([f <= B.CONV_TOL for f in gf]))
        npc = float(np.mean([f <= B.CONV_TOL for f in nf]))
        g_curve.append((_med(gf), gp)); n_curve.append((_med(nf), npc))
        print(f"{bud:>8}{_med(gf):>12.4g}{gp:>12.2f}{_med(nf):>12.4g}{npc:>12.2f}")

    # --- Figures -----------------------------------------------------------
    # Fig A: gradient scale (raw vs logit) + Adam steps/50% range
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
    x = np.arange(len(PARAMS)); w = 0.38
    ax[0].bar(x - w/2, [raw_med[p] for p in PARAMS], w, label="raw", color="#999999")
    ax[0].bar(x + w/2, [logit_med[p] for p in PARAMS], w, label="logit", color="#0072B2")
    ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels(PARAMS)
    ax[0].set_ylabel(r"median $|\partial L/\partial\theta|$"); ax[0].legend()
    ax[0].set_title("init gradient by space")
    ax[1].bar(x, [steps_50[p] for p in PARAMS], color="#D55E00")
    ax[1].axhline(B.MAX_ITERS, ls="--", lw=1, color="black", label=f"budget={B.MAX_ITERS}")
    ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels(PARAMS)
    ax[1].set_ylabel("Adam steps / 50% range"); ax[1].legend()
    ax[1].set_title(f"raw-space reach at lr={lr_ref}")
    fig.savefig(FIGS / "subthreshold_grad_scale.pdf"); plt.close(fig)

    # Fig B: lr sweep heatmap (median final MSE), one panel per transform
    fig, axes = plt.subplots(1, len(TRANSFORMS), figsize=(7.4, 3.0),
                             constrained_layout=True, squeeze=False)
    for ax, tf in zip(axes[0], TRANSFORMS):
        M = np.array([[next(r["med_final"] for r in grid
                            if r["opt"] == o and r["tf"] == tf and r["lr"] == lr)
                       for lr in LRS] for o in OPTIMIZERS])
        im = ax.imshow(np.log10(M), aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(LRS))); ax.set_xticklabels([f"{x:g}" for x in LRS])
        ax.set_yticks(range(len(OPTIMIZERS))); ax.set_yticklabels(OPTIMIZERS)
        ax.set_xlabel("learning rate"); ax.set_title(f"transform={tf}")
        for i in range(len(OPTIMIZERS)):
            for j in range(len(LRS)):
                ax.text(j, i, f"{M[i, j]:.2g}", ha="center", va="center",
                        color="white", fontsize=7)
        fig.colorbar(im, ax=ax, label=r"$\log_{10}$ median MSE")
    fig.savefig(FIGS / "subthreshold_lr_sweep.pdf"); plt.close(fig)

    # Fig C: budget curve, grad-best vs NM
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.9), constrained_layout=True)
    ax[0].plot(BUDGETS, [g[0] for g in g_curve], "o-", color=COLORS["grad"],
               label=f"grad ({best['opt']}, lr={best['lr']})")
    ax[0].plot(BUDGETS, [n[0] for n in n_curve], "^-", color=COLORS["nm"], label="NM")
    ax[0].axhline(B.CONV_TOL, ls=":", lw=1, color="black")
    ax[0].set_yscale("log"); ax[0].set_xlabel("budget (loss evals)")
    ax[0].set_ylabel("median final MSE"); ax[0].legend()
    ax[1].plot(BUDGETS, [g[1] for g in g_curve], "o-", color=COLORS["grad"])
    ax[1].plot(BUDGETS, [n[1] for n in n_curve], "^-", color=COLORS["nm"])
    ax[1].set_xlabel("budget (loss evals)"); ax[1].set_ylabel("P(converged)")
    ax[1].set_ylim(-0.04, 1.04)
    fig.savefig(FIGS / "subthreshold_budget_curve.pdf"); plt.close(fig)

    print(f"\nFigures -> {FIGS}/")
    print(f"\nRecommended GRAD update:\n"
          f"  optimizer={best['opt']!r}, lr={best['lr']}, "
          f"transform={best['tf']}")


if __name__ == "__main__":
    main()
