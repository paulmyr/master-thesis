"""Tune the gradient hyperparameters for each loss in `benchmark_spiking.py`.

The headline question of the benchmark is *how low can each optimizer drive the
loss?*, so gradient descent needs a well-chosen learning rate / surrogate slope /
parameter-transform regime to compete fairly with Nelder-Mead. This script sweeps a
grid at the *same* MAX_ITERS budget (and the *same* cosine LR decay) the main
benchmark uses, scores each config by the best-so-far achieved loss (lower = better),
and prints a ready-to-paste GRAD_HP block.

Cosine LR decay (added to `run_grad`) prevents Adam from finding the basin then
overshooting back out, so best-so-far is now a genuine *held* minimum rather than a
transient dip. We still report the final-vs-best gap as a diagnostic: a large gap
means the config still overshoots and the lr should come down.

Reuses benchmark_spiking for scenarios, cell/loss build, the gradient/NM runners.
Run with `python benchmark_spiking_grad_tuning.py`.
"""

import numpy as np

import benchmark_spiking as B

# --- Tuning config ---------------------------------------------------------

# Mirror the benchmark's actual DELTAS so the tuned HP generalize to the full
# run (a harder/larger-delta-only subset over-selects aggressive lr that then
# overshoots the easy small-delta scenarios).
SUBSET_BASES = ["tonic"]
SUBSET_DELTAS = [0.05, 0.1, 0.2]
SUBSET_DIRS = 5

# Sweep grid (lr, surrogate_slope, transform) per loss. Slope emphasis 5-15
# (Gygax & Zenke 2025: slope 25 gives a ~0.4 mV window); lr spans low-to-moderate
# since cosine decay means the *initial* lr sets the early step size.
_LR = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
_SLOPE = [5.0, 7.5, 10.0, 15.0]
GRID = {
    "ttfs_rate":  dict(lr=_LR, slope=_SLOPE, transform=[True]),
    "van_rossum": dict(lr=_LR, slope=_SLOPE, transform=[True]),
    "guarino":    dict(lr=_LR, slope=_SLOPE, transform=[True]),
    "soft_dtw":   dict(lr=_LR, slope=_SLOPE, transform=[True]),
}


def _med(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def main():
    # Restrict the scenario set to a small, representative subset.
    B.BASES = SUBSET_BASES
    B.DELTAS = SUBSET_DELTAS
    B.N_DIRS = SUBSET_DIRS
    scenarios = B.generate_scenarios()
    print(f"Tuning subset: {len(scenarios)} scenarios "
          f"(bases={SUBSET_BASES}, deltas={SUBSET_DELTAS}, {SUBSET_DIRS} dirs)\n")

    best_hp = {}
    for loss_name in B.LOSSES:
        g = GRID[loss_name]
        # NM baseline (same budget): achieved loss = min over its eval curve.
        nm_losses = [min(B.run_nm(sc["init"], sc["trace"], loss_name)[1])
                     for sc in scenarios]
        nm_med = _med(nm_losses)
        print("=" * 72)
        print(f"LOSS = {loss_name}   (NM median achieved loss = {nm_med:.4g}, "
              f"budget = {B.MAX_ITERS})")
        print("=" * 72)
        print(f"{'lr':>8}{'slope':>8}{'transform':>11}{'med loss':>12}"
              f"{'med gap':>10}{'win% vs NM':>12}")
        results = []
        for lr in g["lr"]:
            for slope in g["slope"]:
                for tf in g["transform"]:
                    hp = dict(lr=lr, surrogate_slope=slope, transform=tf)
                    curves = [B.run_grad(sc["init"], sc["trace"], loss_name, hp)[1]
                              for sc in scenarios]
                    losses = [min(c) for c in curves]            # best-so-far achieved
                    gaps = [c[-1] - min(c) for c in curves]      # overshoot diagnostic
                    med = _med(losses)
                    gap = _med(gaps)
                    win = float(np.mean([gl < nl
                                         for gl, nl in zip(losses, nm_losses)]))
                    results.append((med, hp))
                    print(f"{lr:>8}{slope:>8}{str(tf):>11}{med:>12.4g}"
                          f"{gap:>10.3g}{100 * win:>11.0f}%")
        results.sort(key=lambda r: r[0])  # lowest median achieved loss first
        best_hp[loss_name] = results[0][1]
        print(f"  BEST: {results[0][1]}  (med loss = {results[0][0]:.4g})\n")

    print("=" * 72)
    print("Paste into benchmark_spiking.py:")
    print("=" * 72)
    print("GRAD_HP = {")
    for loss_name in B.LOSSES:
        hp = best_hp[loss_name]
        print(f"    {loss_name!r:<14}: dict(lr={hp['lr']}, "
              f"surrogate_slope={hp['surrogate_slope']}, "
              f"transform={hp['transform']}),")
    print("}")


if __name__ == "__main__":
    main()
