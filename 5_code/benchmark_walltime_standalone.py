"""
Best-case Brian2 walltime: cpp_standalone (compiled C++, no Python in the loop).

`benchmark_walltime.py` runs Brian2 in runtime/Cython mode, which keeps the
integration loop in Python -- for a single point neuron that per-timestep Python
overhead dominates, i.e. Brian2's *worst* case. The Brian2 docs
(https://brian2.readthedocs.io/en/stable/user/computation.html) recommend
cpp_standalone for "impressive speedups": it generates and compiles a standalone
C++ program with no Python in the loop. This script gives Brian2 that best case
so the Jaxley-vs-Brian2 claim cannot be dismissed as an artefact of Brian2's slow
default mode.

Fairness choices baked into the measurement:
  - Matched numerics with benchmark_walltime.py: forward Euler, same dt, same
    equations/parameters (spike-count guard against Jaxley).
  - Brian2's OWN default optimized build flags (-O3 -ffast-math -march=native, see
    prefs.codegen.cpp.extra_compile_args_gcc) are used UNMODIFIED -- the most
    defensible "best case": we did not hand-tune anything Brian2 doesn't already do.
  - The one-time codegen+compile (device.build, the analog of JAX JIT) is timed
    separately from per-run integration.
  - Two per-run numbers are logged because they measure different things:
      * binary_loop_s -- the pure integration loop time the compiled C++ program
        reports for ITSELF (results/last_run_info.txt). This is the apples-to-
        apples analog of Jaxley's warm in-process compute time: both exclude
        process launch and result serialisation.
      * run_wall_s -- end-to-end wall time of device.run(), which re-executes the
        binary from scratch and therefore ALSO pays OS process spawn + result
        file I/O on every call (~10 ms here, dwarfing the ~0.6 ms integration).
        This is what a naive one-shot standalone user pays per simulation.
    Reporting both keeps the comparison honest in both directions: binary_loop_s
    is Brian2's fairest number, run_wall_s is its realistic per-call number.
  - store()/restore() is unsupported in standalone; repeats use device.run(),
    which re-initialises the binary each call (identical initial condition every
    run). The first device.run() is discarded as warm-up (cold process/dylib
    cache), matching how the Jaxley baseline discards its JIT-compile run.

Environment note: on this machine bare `c++` resolves to a broken x86_64 *Linux*
ELF binary in /usr/local/bin, so Brian2's standalone Makefile (which invokes
$(CXX), defaulting to `c++`) is pointed at Apple clang via the CXX/CC env vars
below. Adjust COMPILER_CXX / COMPILER_CC for your machine; leave as None to use
whatever `c++`/`cc` your PATH provides.
"""

import os
import shutil

# Must be set before Brian2 picks a compiler for the standalone Makefile.
COMPILER_CXX = "/usr/bin/clang++"  # None -> use PATH default
COMPILER_CC = "/usr/bin/clang"
if COMPILER_CXX and shutil.which(COMPILER_CXX):
    os.environ["CXX"] = COMPILER_CXX
if COMPILER_CC and shutil.which(COMPILER_CC):
    os.environ["CC"] = COMPILER_CC

import time

import numpy as np

# Reuse the matched config + Jaxley baseline from the runtime-mode benchmark so
# the two scripts stay in lockstep (same params, dt, durations, stimulus).
from benchmark_walltime import (DT_MS, DURATIONS_MS, N_REPEATS, PARAMS,
                                STIM_CURRENT_PA, bench_jaxley, stim_window)

BUILD_DIR = "standalone_build"
OUT_CSV = "results_walltime_standalone.csv"
REPEATS_CSV = "results_walltime_standalone_repeats.csv"


def read_binary_loop_s(directory: str) -> float:
    """Pure integration-loop seconds the compiled binary reports for itself."""
    path = os.path.join(directory, "results", "last_run_info.txt")
    with open(path) as f:
        return float(f.read().split()[0])


def bench_standalone(t_max_ms: float):
    """(build_s, run_wall_times, binary_loop_times, n_spikes) for cpp_standalone."""
    from brian2 import (BrianLogger, NeuronGroup, SpikeMonitor, TimedArray,
                        defaultclock, device, mV, ms, nS, pA, pF, run,
                        set_device)

    BrianLogger.suppress_name("resolution_conflict")
    # Build without running so codegen+compile is timed on its own; reinit lets
    # us build a fresh standalone project per duration in one process.
    set_device("cpp_standalone", build_on_run=False)
    device.reinit()
    device.activate(build_on_run=False)
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
        method="euler", namespace=namespace,
    )
    neuron.v = PARAMS["E_L"] * mV
    neuron.w = 0 * pA
    spikes = SpikeMonitor(neuron)
    run(t_max_ms * ms)  # records the run; build_on_run=False defers compilation

    build_dir = f"{BUILD_DIR}_{int(t_max_ms)}"
    t0 = time.perf_counter()
    device.build(run=False, directory=build_dir, clean=True)
    build_s = time.perf_counter() - t0

    # First run is cold (process spawn + dylib cache); discard it as warm-up.
    device.run()
    run_wall, binary_loop = [], []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        device.run()
        run_wall.append(time.perf_counter() - t0)
        binary_loop.append(read_binary_loop_s(build_dir))

    n_spikes = int(spikes.num_spikes)
    return build_s, run_wall, binary_loop, n_spikes


def main() -> None:
    agg_rows, repeat_rows = [], []
    print(f"{'t_max':>8} {'simulator':>18} {'metric':>14} "
          f"{'mean_ms':>10} {'std_ms':>9} {'speedup':>8} {'spikes':>8}")

    for t_max in DURATIONS_MS:
        jx_compile, jx_times, jx_n = bench_jaxley(t_max)
        jx_mean = float(np.mean(jx_times))
        agg_rows.append((t_max, DT_MS, "jaxley", "compute", jx_mean,
                         float(np.std(jx_times)), jx_compile, jx_n, N_REPEATS))
        for i, t in enumerate(jx_times):
            repeat_rows.append((t_max, DT_MS, "jaxley", "compute", i, float(t), jx_n))
        print(f"{t_max:8.0f} {'jaxley':>18} {'compute':>14} "
              f"{jx_mean * 1e3:10.3f} {np.std(jx_times) * 1e3:9.3f} {'--':>7} {jx_n:>8}")

        build_s, run_wall, binary_loop, br_n = bench_standalone(t_max)
        flag = "" if abs(jx_n - br_n) <= 1 else "  <-- SPIKE MISMATCH"
        # binary_loop = fairest (compute vs compute); run_wall = realistic per-call.
        for metric, times, one_time in (
            ("binary_loop", binary_loop, build_s),
            ("run_wall", run_wall, build_s),
        ):
            mean = float(np.mean(times))
            agg_rows.append((t_max, DT_MS, "brian2_standalone", metric, mean,
                             float(np.std(times)), one_time, br_n, N_REPEATS))
            for i, t in enumerate(times):
                repeat_rows.append(
                    (t_max, DT_MS, "brian2_standalone", metric, i, float(t), br_n))
            print(f"{t_max:8.0f} {'brian2_standalone':>18} {metric:>14} "
                  f"{mean * 1e3:10.3f} {np.std(times) * 1e3:9.3f} "
                  f"{mean / jx_mean:7.1f}x {br_n:>8}{flag if metric=='binary_loop' else ''}")

    with open(OUT_CSV, "w") as f:
        f.write("t_max_ms,dt_ms,simulator,metric,mean_s,std_s,one_time_s,spikes,n_repeats\n")
        for r in agg_rows:
            f.write(",".join(str(x) for x in r) + "\n")
    with open(REPEATS_CSV, "w") as f:
        f.write("t_max_ms,dt_ms,simulator,metric,repeat,time_s,spikes\n")
        for r in repeat_rows:
            f.write(",".join(str(x) for x in r) + "\n")

    print(f"\nWrote {OUT_CSV} (mean/std per sim+metric; one_time_s = JIT/compile)")
    print(f"Wrote {REPEATS_CSV} (one row per repeat, for error bars)")
    print("metric legend: jaxley 'compute' & brian2 'binary_loop' are the fair "
          "compute-vs-compute pair; 'run_wall' adds Brian2's per-call process "
          "spawn + result I/O.")


if __name__ == "__main__":
    main()
