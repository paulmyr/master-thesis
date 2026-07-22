"""
Walltime benchmark: AdEx in Jaxley vs Brian2 with MATCHED numerical integration.

For the comparison to be scientifically fair both simulators must integrate the
*same* ODE with the *same* scheme and the *same* time step:

  - Scheme: forward (explicit) Euler in both.
      Brian2:  method="euler".
      Jaxley:  solver="fwd_euler"  (NOTE: Jaxley defaults to "bwd_euler", an
               implicit scheme Brian2 is not using — passing fwd_euler is what
               makes this apples-to-apples).
  - Time step: same dt in both (defaultclock.dt / delta_t).
  - Equations & parameters: identical (already verified <1 ms spike timing
    against Brian2 elsewhere; we re-assert equal spike counts here as a guard).

We separate one-time cost (JAX JIT compile / Brian2 code generation) from the
per-run integration walltime, which is the quantity being compared. Jaxley runs
are timed with block_until_ready() to defer async dispatch into the timed region;
Brian2 runs reset state with store()/restore() so each timed run integrates from
the same initial condition.

Every repeat is timed and logged (long format, one row per repeat) so downstream
analysis can compute mean +/- std error bars rather than only the min. Brian2 is
additionally swept over all its integration schemes that support the (nonlinear)
AdEx equations so the fastest scheme can be identified; the linear-only schemes
(exact/linear/exponential_euler) and any GSL solver that is not installed are
skipped automatically. Jaxley stays on forward Euler as the matched baseline.
"""

import time

import jax
import jaxley as jx
import numpy as np

from ADoptEX.core.parameters import DEFAULT_PARAMS
from ADoptEX.core.simulation import create_adex_cell

# --- Configuration -----------------------------------------------------------
PARAMS = dict(DEFAULT_PARAMS)
DT_MS = 0.025
STIM_DELAY_MS = 50.0
STIM_CURRENT_PA = 500.0
DURATIONS_MS = [200.0, 500.0, 1000.0, 2000.0, 5000.0]  # total sim time sweep
N_REPEATS = 20
# Brian2 integration schemes to sweep. The linear-only schemes (exact/linear/
# exponential_euler) do not support the nonlinear AdEx exp term and GSL solvers
# need the gsl package; both are skipped at runtime if they raise.
BRIAN_METHODS = ["euler", "heun", "milstein", "rk2", "rk4"]
OUT_CSV = "results_walltime.csv"  # aggregate (mean/std/min per sim+method)
REPEATS_CSV = "results_walltime_repeats.csv"  # long-format per-repeat timings


def stim_window(t_max_ms: float) -> tuple[float, float]:
    """Stimulus (delay, duration): current on from delay until 50 ms before end."""
    duration = t_max_ms - STIM_DELAY_MS - 50.0
    return STIM_DELAY_MS, duration


def bench_jaxley(t_max_ms: float) -> tuple[float, list[float], int]:
    """Return (compile_s, per_run_times, n_spikes) for forward-Euler Jaxley."""
    delay, dur = stim_window(t_max_ms)
    cell = create_adex_cell(PARAMS, use_surrogate=False, v_init=PARAMS["E_L"])
    cell.stimulate(
        jx.step_current(delay, dur, STIM_CURRENT_PA / 1000.0, DT_MS, t_max=t_max_ms)
    )

    @jax.jit
    def run():
        return jx.integrate(cell, delta_t=DT_MS, t_max=t_max_ms, solver="fwd_euler")

    t0 = time.perf_counter()
    out = run()
    out.block_until_ready()
    compile_s = time.perf_counter() - t0

    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        out = run()
        out.block_until_ready()
        times.append(time.perf_counter() - t0)

    n_spikes = int((np.asarray(out[2]) > 0.5).sum())
    return compile_s, times, n_spikes


def bench_brian2(t_max_ms: float, method: str) -> tuple[float, list[float], int]:
    """Return (codegen_s, per_run_times, n_spikes) for Brian2 with `method`."""
    from brian2 import (BrianLogger, Network, NeuronGroup, SpikeMonitor,
                        TimedArray, defaultclock, mV, ms, nS, pA, pF, prefs)

    BrianLogger.suppress_name("resolution_conflict")
    prefs.codegen.target = "cython"
    defaultclock.dt = DT_MS * ms

    delay, dur = stim_window(t_max_ms)
    n = int(round(t_max_ms / DT_MS))
    current = np.zeros(n)
    current[int(delay / DT_MS) : int((delay + dur) / DT_MS)] = STIM_CURRENT_PA
    I_ta = TimedArray(current * pA, dt=DT_MS * ms)

    eqs = """
    dv/dt = (g_L*(E_L-v) + g_L*delta_T*exp((v-v_T)/delta_T) - w + I_ta(t))/C_m : volt
    dw/dt = (a*(v-E_L) - w)/tau_w : amp
    """
    namespace = dict(
        C_m=PARAMS["C_m"] * pF, g_L=PARAMS["g_L"] * nS, E_L=PARAMS["E_L"] * mV,
        v_T=PARAMS["v_T"] * mV, delta_T=PARAMS["delta_T"] * mV,
        v_reset=PARAMS["v_reset"] * mV, v_threshold=PARAMS["v_threshold"] * mV,
        tau_w=PARAMS["tau_w"] * ms, a=PARAMS["a"] * nS, b=PARAMS["b"] * pA, I_ta=I_ta,
    )
    neuron = NeuronGroup(
        1, eqs, threshold="v>v_threshold", reset="v=v_reset; w+=b",
        method=method, namespace=namespace,
    )
    neuron.v = PARAMS["E_L"] * mV
    neuron.w = 0 * pA
    spikes = SpikeMonitor(neuron)
    net = Network(neuron, spikes)
    net.store()

    t0 = time.perf_counter()
    net.run(t_max_ms * ms)
    codegen_s = time.perf_counter() - t0
    n_spikes = int(spikes.num_spikes)

    times = []
    for _ in range(N_REPEATS):
        net.restore()
        t0 = time.perf_counter()
        net.run(t_max_ms * ms)
        times.append(time.perf_counter() - t0)

    return codegen_s, times, n_spikes


def summarize(t_max, sim, method, times, compile_s, n_spikes):
    """(aggregate row, list of per-repeat rows) for one benchmarked config."""
    arr = np.asarray(times)
    agg = (t_max, DT_MS, sim, method, float(arr.mean()), float(arr.std()),
           float(arr.min()), compile_s, n_spikes, N_REPEATS)
    reps = [(t_max, DT_MS, sim, method, i, float(t), n_spikes)
            for i, t in enumerate(times)]
    return agg, reps


def main() -> None:
    agg_rows, repeat_rows = [], []
    print(f"{'t_max':>8} {'simulator':>10} {'method':>16} "
          f"{'mean_ms':>10} {'std_ms':>9} {'speedup':>8} {'spikes':>8}")
    for t_max in DURATIONS_MS:
        jx_compile, jx_times, jx_n = bench_jaxley(t_max)
        agg, reps = summarize(t_max, "jaxley", "fwd_euler", jx_times, jx_compile, jx_n)
        agg_rows.append(agg)
        repeat_rows.extend(reps)
        jx_mean = float(np.mean(jx_times))
        print(f"{t_max:8.0f} {'jaxley':>10} {'fwd_euler':>16} "
              f"{jx_mean * 1e3:10.3f} {np.std(jx_times) * 1e3:9.3f} "
              f"{'--':>7} {jx_n:>8}")

        for method in BRIAN_METHODS:
            try:
                br_codegen, br_times, br_n = bench_brian2(t_max, method)
            except Exception as e:  # scheme unsupported for these eqs / no GSL
                print(f"{t_max:8.0f} {'brian2':>10} {method:>16}   "
                      f"skipped ({type(e).__name__})")
                continue
            agg, reps = summarize(t_max, "brian2", method, br_times, br_codegen, br_n)
            agg_rows.append(agg)
            repeat_rows.extend(reps)
            # Guard that Brian2 integrates the same model as Jaxley. A ±1 spike
            # difference is expected at threshold-crossing boundaries (the
            # implementations agree to <1 ms spike timing); larger gaps mean the
            # setups have drifted apart and the walltimes are not comparable.
            flag = "" if abs(jx_n - br_n) <= 1 else "  <-- SPIKE MISMATCH"
            br_mean = float(np.mean(br_times))
            print(f"{t_max:8.0f} {'brian2':>10} {method:>16} "
                  f"{br_mean * 1e3:10.3f} {np.std(br_times) * 1e3:9.3f} "
                  f"{br_mean / jx_mean:7.1f}x {br_n:>8}{flag}")

    agg_header = ("t_max_ms,dt_ms,simulator,method,mean_s,std_s,min_s,"
                  "compile_s,spikes,n_repeats")
    with open(OUT_CSV, "w") as f:
        f.write(agg_header + "\n")
        for r in agg_rows:
            f.write(",".join(str(x) for x in r) + "\n")

    rep_header = "t_max_ms,dt_ms,simulator,method,repeat,time_s,spikes"
    with open(REPEATS_CSV, "w") as f:
        f.write(rep_header + "\n")
        for r in repeat_rows:
            f.write(",".join(str(x) for x in r) + "\n")

    print(f"\nWrote {OUT_CSV} (mean/std/min over {N_REPEATS} repeats per sim+method)")
    print(f"Wrote {REPEATS_CSV} (one row per repeat, for error bars)")


if __name__ == "__main__":
    main()