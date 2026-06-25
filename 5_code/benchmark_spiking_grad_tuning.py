"""Tune the gradient hyperparameters for the ttfs_rate loss in `benchmark_spiking.py`.

The headline question of the benchmark is *how low can each optimizer drive the
ttfs_rate loss?*, so gradient descent needs a well-chosen learning rate / surrogate
slope / parameter-transform regime to compete fairly with Nelder-Mead. This script
sweeps a short grid at the *same* MAX_ITERS budget the main benchmark uses, scores
each config by the achieved loss (min over the optimization curve, lower = better),
and prints a ready-to-paste GRAD_HP line.

Reuses benchmark_spiking for scenarios, cell/loss build, the gradient/NM runners.
Run with `python benchmark_spiking_grad_tuning.py`.
"""

import numpy as np

import benchmark_spiking as B

# --- Tuning config ---------------------------------------------------------

SUBSET_BASES = ["tonic"]
SUBSET_DELTAS = [0.2]
SUBSET_DIRS = 3

# Sweep grid (lr, surrogate_slope, transform) for the ttfs_rate loss.
GRID = {
    "ttfs_rate": dict(lr=[0.2, 0.5, 1.0], slope=[5.0, 10.0], transform=[False, True]),
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
              f"{'win% vs NM':>12}")
        results = []
        for lr in g["lr"]:
            for slope in g["slope"]:
                for tf in g["transform"]:
                    hp = dict(lr=lr, surrogate_slope=slope, transform=tf)
                    losses = [min(B.run_grad(sc["init"], sc["trace"], loss_name, hp)[1])
                              for sc in scenarios]
                    med = _med(losses)
                    win = float(np.mean([gl < nl
                                         for gl, nl in zip(losses, nm_losses)]))
                    results.append((med, hp))
                    print(f"{lr:>8}{slope:>8}{str(tf):>11}{med:>12.4g}"
                          f"{100 * win:>11.0f}%")
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
