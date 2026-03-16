# AdEx Unit Conversion: Analysis and Paths Forward

**Date:** 2025-03-13
**Context:** Analysis of how the AdEx channel implementation maps between Naud et al.'s
point-neuron units and Jaxley's compartmental (density) unit system.

## The Problem

The Naud et al. AdEx model is a **point neuron** — it has no spatial extent. All parameters
are in absolute units (pF, nS, pA). Jaxley is a **compartmental simulator** where everything
is expressed as densities per membrane area (μF/cm², S/cm², mA/cm²). The current implementation
mixes both systems in a way that happens to work but is confusing and fragile.

## How the Current Implementation Works

### Parameter Table: Paper vs Jaxley Units

| Parameter | Paper (Naud) | Jaxley native | Current code uses |
|-----------|-------------|---------------|-------------------|
| C_m | 200 pF | μF/cm² | **200** (raw number, set as μF/cm²) |
| g_L | 10 nS | S/cm² | **10** (raw number, set as S/cm²) |
| a | 2 nS | S/cm² | **2** (raw number) |
| b | 0 pA | mA/cm² | **0** (raw number) |
| w | pA | mA/cm² | dimensionless (matches pA numerically) |
| E_L, v_T, etc. | mV | mV | mV (no conversion needed) |
| tau_w | ms | ms | ms (no conversion needed) |
| I_ext | 500 pA | nA (via stimulate) | **100 nA** (= 500 × 200 / 1000) |

### Architecture: Two Parallel Current Pathways

The AdEx channel has a critical design choice: it handles voltage integration **internally**
in `update_states()`, and `compute_current()` returns **zero**.

This creates two independent pathways for how currents affect voltage:

**Pathway 1 — Internal (bypasses Jaxley's pipeline):**
```
AdEx.update_states():
    i_leak = g_L * (v - E_L)           # raw numbers: 10 * (v - (-70))
    i_exp  = g_L * delta_T * exp(...)   # raw numbers
    i_adapt = w                          # raw number
    dv = (-i_leak + i_exp - i_adapt) / C_m   # divided by 200
    v = v + dt * dv
```
Since `compute_current()` returns 0, Jaxley's density pipeline never touches these currents.
The ratio g_L/C_m = 10/200 = 0.05 /ms matches the paper's 10 nS / 200 pF exactly.
**No geometry dependence whatsoever.**

**Pathway 2 — External current (goes through Jaxley's pipeline):**
```
cell.stimulate(I_nA)
  → convert_point_process_to_distributed(I_nA, area_μm²)
    → I_nA / area_μm² × 100,000  →  i_ext in μA/cm²
  → Jaxley solver: dv += i_ext / capacitance × dt
```
This pathway **does** depend on geometry and capacitance.

### Why the Geometry Hack Exists

The geometry is set via `geometry_for_capacitance(C_m)`, which computes a cylindrical cell
whose surface area satisfies: `area_cm² = C_m_pF × 1e-6 / 1.0` (assuming 1.0 μF/cm²
specific capacitance). For C_m = 200 pF → area = 2×10⁻⁴ cm² = 20,000 μm².

**But then `cell.set("capacitance", 200)` overrides the specific capacitance to 200 μF/cm²!**
This makes the total capacitance = 200 × 2×10⁻⁴ = 0.04 μF = 40,000,000 pF — physically
nonsensical. It only works because:

1. Inside the channel: C_m is read as raw 200, matching the paper
2. For external current: the geometry and the I_nA conversion are coordinated

### The Current Conversion: `I_nA = I_pA × C_m / 1000`

This is **not** a simple pA-to-nA conversion (which would be ÷1000). It also multiplies by
C_m = 200, giving I_nA = 500 × 200 / 1000 = 100 nA.

Tracing through the pipeline:
```
i_ext_density = I_nA / area_μm² × 100,000
             = (I_pA × C_m_init / 1000) / (C_m_init × 100) × 100,000
               ^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^
               I_nA formula               area = C_m_init × 100 μm²

             = I_pA × (C_m_init / C_m_init) × (100,000 / 100,000)
             = I_pA    (numerically, in μA/cm²)
```

**The initial C_m cancels perfectly between the I_nA formula and the area.**
Then Jaxley's solver computes: `dv_ext = I_pA / C_m_trainable` — which is exactly the
paper equation, even if C_m changes during training.

### Why g_L Does NOT Need Geometry Transformation

g_L is used exclusively in Pathway 1 (internal). It never enters Jaxley's density pipeline
because `compute_current()` returns 0. The channel uses g_L = 10 as a raw number in
`dv = (-g_L*(v-E_L) + ...) / C_m`, and the ratio 10/200 matches the paper. The same applies
to `a`, `b`, `w`, and all other internal parameters. **There is no bug.**

### Training C_m: Does It Break?

**No.** As shown above, i_ext_density simplifies to just I_pA (the initial C_m cancels).
So `dv_ext = I_pA / C_m_trainable`, which is correct for any C_m value. The gradient
`∂(dv_ext)/∂C_m = -I_pA / C_m²` is also correct. Trained C_m values are directly
interpretable as picofarads.

### Why It's Confusing and Fragile

1. `cell.set("capacitance", 200)` looks like 200 μF/cm² but means 200 pF
2. `I_nA = I × C_m / 1000` looks like a bug but is a carefully balanced compensation
3. The geometry has no physical meaning — it exists only to make the current conversion work
4. The correctness depends on **three things being perfectly coordinated**: the geometry
   function, the I_nA formula, and the capacitance value. Change any one and it silently breaks.
5. Trained parameters are in paper units but stored in Jaxley fields labeled as density units

---

## Approach A: Clamp (Point-Neuron, Single-Compartment)

### Concept

Handle **everything** inside the channel using paper units. Pass external current as a
clamped channel state instead of going through Jaxley's `stimulate()` pipeline.
Cell geometry becomes completely irrelevant.

### Changes to AdEx Channel (`adex.py`)

Add `I_ext` as a channel state:

```python
# In __init__:
self.channel_states = {
    f"{prefix}_w": 0.0,
    f"{prefix}_spikes": False,
    f"{prefix}_I_ext": 0.0,  # External current in pA (paper units)
}
```

Use it in `update_states()`:

```python
# In update_states:
I_ext = states[f"{prefix}_I_ext"]

# Voltage update — everything in paper units
dv = (-i_leak + i_exp - i_adapt + I_ext) / C_m
v = v + dt * dv
```

### Changes to Notebook / Simulation Setup

```python
# BEFORE (confusing):
radius_um, length_um, area_cm2 = geometry_for_capacitance(params["C_m"])
cell.set("radius", radius_um)
cell.set("length", length_um)
cell.set("capacitance", params["C_m"])
I_nA = params["I"] * params["C_m"] / 1000.0
cell.stimulate(jx.step_current(0.0, duration_ms, I_nA, dt_ms, t_max=t_max_ms))

# AFTER (clear):
cell.set("capacitance", params["C_m"])  # Raw paper value, used inside channel
# No geometry setup needed
# Build stimulus in paper units (pA):
n_steps = int(t_max_ms / dt_ms) + 1
current_pA = jnp.zeros(n_steps)
stim_steps = int(duration_ms / dt_ms)
current_pA = current_pA.at[:stim_steps].set(params["I"])  # e.g. 500 pA
cell.clamp("AdEx_I_ext", current_pA)
# No cell.stimulate() call
```

### What Gets Eliminated

- `geometry_for_capacitance()` function — deleted entirely
- `cell.set("radius", ...)` and `cell.set("length", ...)` — unnecessary
- `I_nA = I × C_m / 1000` — replaced by direct pA values
- All implicit coupling between geometry, current scaling, and capacitance

### How It Works

1. `cell.clamp("AdEx_I_ext", array)` tells Jaxley to set the state to the given value at
   each timestep
2. The channel reads `states["AdEx_I_ext"]` = 500.0 (pA) during the stimulus window
3. `dv = (... + 500) / 200` = paper equation exactly
4. `compute_current()` still returns 0, so Jaxley's solver adds nothing on top
5. No external current goes through Jaxley's density pipeline at all

### Trained Parameters

Directly in paper units (pF, nS, pA, mV, ms). Transferable to Brian2 or any other simulator
without conversion.

### Limitations

- **Single-compartment only.** Synaptic input in a network arrives through Jaxley's
  `compute_current` → density pipeline, which this approach bypasses entirely.
- Cannot use `jx.step_current()` or other Jaxley stimulus utilities (must build arrays manually,
  though a small helper function suffices).
- If Jaxley changes how `clamp` interacts with `update_states`, this could break.

### Applicability for Thesis

This approach is well-suited. The thesis fits a point neuron (AdEx) against single traces.
There are no multi-compartment cells, no networks, no synapses. The clamp approach is honest
about what we're doing: treating AdEx as a point neuron in paper units.

---

## Approach B: Density Conversion (Network-Compatible)

### Concept

Convert all paper parameters to Jaxley's native density units at setup time, given a chosen
cell geometry. The channel works entirely in density units and integrates properly with
Jaxley's current pipeline for synapses, stimuli, and multi-compartment models.

### Unit Conversion Functions

```python
def naud_to_density(params_naud, area_cm2):
    """Convert Naud et al. absolute params → Jaxley density params.

    Extensive quantities (scale with membrane area) are divided by area.
    Intensive quantities (voltages, time constants) are unchanged.

    Args:
        params_naud: dict with keys C_m (pF), g_L (nS), a (nS), b (pA),
                     E_L (mV), v_T (mV), delta_T (mV), tau_w (ms),
                     v_reset (mV), v_threshold (mV), I (pA)
        area_cm2: membrane surface area in cm²

    Returns:
        dict with Jaxley density units
    """
    return {
        # Extensive → density (divide by area, convert SI prefix)
        "capacitance": params_naud["C_m"] * 1e-6 / area_cm2,     # pF → μF/cm²
        "AdEx_g_L":    params_naud["g_L"] * 1e-9 / area_cm2 * 1e3,  # nS → mS/cm²
        "AdEx_a":      params_naud["a"]   * 1e-9 / area_cm2 * 1e3,  # nS → mS/cm²
        "AdEx_b":      params_naud["b"]   * 1e-9 / area_cm2,        # pA → nA → nA/cm²
        # ^^^ NOTE: b and w units need careful verification against Jaxley's
        # expected current unit (mA/cm² vs μA/cm²). See implementation notes below.

        # Intensive → unchanged
        "AdEx_E_L":        params_naud["E_L"],         # mV
        "AdEx_v_T":        params_naud["v_T"],          # mV
        "AdEx_delta_T":    params_naud["delta_T"],      # mV
        "AdEx_v_threshold": params_naud["v_threshold"],  # mV
        "AdEx_v_reset":    params_naud["v_reset"],       # mV
        "AdEx_tau_w":      params_naud["tau_w"],         # ms
    }


def density_to_naud(params_density, area_cm2):
    """Convert trained Jaxley density params back to Naud absolute units."""
    return {
        "C_m": params_density["capacitance"] / 1e-6 * area_cm2,        # μF/cm² → pF
        "g_L": params_density["AdEx_g_L"] / 1e3 / 1e-9 * area_cm2,    # mS/cm² → nS
        "a":   params_density["AdEx_a"]   / 1e3 / 1e-9 * area_cm2,    # mS/cm² → nS
        "b":   params_density["AdEx_b"]   / 1e-9 * area_cm2,           # nA/cm² → pA
        "E_L":        params_density["AdEx_E_L"],
        "v_T":        params_density["AdEx_v_T"],
        "delta_T":    params_density["AdEx_delta_T"],
        "v_threshold": params_density["AdEx_v_threshold"],
        "v_reset":    params_density["AdEx_v_reset"],
        "tau_w":      params_density["AdEx_tau_w"],
    }
```

### Changes to AdEx Channel

The channel math in `update_states()` stays **identical**. The equation
`dv = (-g_L*(v-E_L) + g_L*delta_T*exp(...) - w) / C_m` doesn't change — only the
magnitudes of the numbers change because they're now in density units. The ratios
are preserved (that's the point of dividing everything by the same area).

**No structural changes to `adex.py` needed.** The channel already reads `params["capacitance"]`
and channel params, computes dv, and updates v. It doesn't care whether the numbers are in
absolute or density units — only the ratios matter, and those are preserved by the conversion.

### Changes to Simulation Setup

```python
# Choose a geometry (physically motivated or arbitrary — just be consistent)
area_cm2 = 2e-4  # or from a real morphology

# Convert paper params to density
density_params = naud_to_density(NAUD_PARAMETERS["tonic"], area_cm2)

# Set up cell with proper geometry for that area
# (radius and length chosen to produce the target area)
cell = jx.Cell()
cell.set("radius", radius_for_area(area_cm2))
cell.set("length", length_for_area(area_cm2))

# Set density parameters
for key, val in density_params.items():
    cell.set(key, val)

# External current: just use Jaxley's standard stimulate() in nA
I_nA = params["I"] * 1e-3   # pA → nA (simple, honest conversion)
cell.stimulate(jx.step_current(0.0, duration_ms, I_nA, dt_ms, t_max=t_max_ms))
```

### How External Current Works Correctly

With proper density units:
```
i_ext_density = I_nA / area_μm² × 100,000     (Jaxley's conversion, result in μA/cm²)
dv_ext = i_ext_density / capacitance_density   (Jaxley's solver)
```

This gives the correct `dv_ext = I_pA / C_m_pF` because the area factors cancel:
```
dv_ext = (I_pA × 1e-3) / (area_cm2 × 1e8) × 1e5 / (C_m_pF × 1e-6 / area_cm2)
       = I_pA / C_m_pF  (after simplification)
```

### How Synaptic Input Works

Synaptic currents arrive as current densities through `compute_current()` of synapse channels.
Jaxley sums them and divides by capacitance. Since the AdEx channel now uses proper density
capacitance, this division is correct. The channel's `compute_current()` still returns 0
(the internal dynamics are still in `update_states`), but external currents from synapses
are handled correctly by Jaxley's solver adding `i_syn / cm` to the voltage.

### Trained Parameters

In density units. Must be converted back via `density_to_naud()` for reporting or use in
other simulators. The conversion is a simple linear scaling that doesn't affect the
optimization landscape (all extensive parameters are multiplied by the same `1/area`).

### Implementation Notes and Pitfalls

1. **Current units need verification.** Jaxley channels declare `current_is_in_mA_per_cm2`.
   The exact unit chain for b, w, and how they interact with the voltage equation needs
   careful verification against Jaxley's HH channels as a reference. Check what units
   `convert_point_process_to_distributed` actually produces (the docstring says μA/cm²,
   but the flag says mA/cm²). This is critical to get right.

2. **The adaptation equation must be consistent.** If g_L is in mS/cm² and C_m is in μF/cm²,
   then `a` should also be in mS/cm² so that `a*(V-E_L)` has the same units as w. And b
   must be in the same units as w. Trace through the full equation with units to verify.

3. **Area choice is arbitrary** for a point neuron. Any area gives the same dynamics. But
   the choice affects the numerical magnitudes of density parameters, which could affect
   optimizer behavior (learning rates, gradient magnitudes). A "nice" area that gives
   capacitance ≈ 1.0 μF/cm² (the biophysical standard) would be:
   `area = C_m_pF × 1e-6 / 1.0 = 200 × 1e-6 = 2×10⁻⁴ cm²`.

4. **PARAM_BOUNDS in `parameters.py` would need density-unit equivalents**, or the bounds
   must be converted alongside the parameters.

---

## Decision Guide

| Criterion | Approach A (Clamp) | Approach B (Density) |
|-----------|-------------------|---------------------|
| Complexity | Simple | Moderate (unit conversion layer) |
| Clarity | Very clear — all paper units | Clear once you understand the conversion |
| Network compatible | No | Yes |
| Synaptic input | Not supported | Works natively |
| Multi-compartment | Not supported | Works natively |
| Trained param format | Paper units (pF, nS) | Density units (need back-conversion) |
| Risk of unit bugs | Low (no conversions) | Moderate (must verify current units) |
| Thesis suitability | Excellent | Overkill for current scope |
| Future extensibility | Would need rewrite | Ready for networks |

### Recommendation

Implement **Approach A** for the thesis. It matches the scope (single-compartment point-neuron
fitting), produces directly interpretable parameters, and eliminates all unit confusion.

Mention Approach B in the thesis discussion as the path to network-scale simulations. The
conversion is a linear rescaling that preserves the optimization landscape, so all findings
about loss functions and surrogate gradients transfer directly.

---

## Files to Modify

### For Approach A
- `3_jaxley/jaxley/channels/non_capacitive/adex.py` — add `I_ext` state, use in `update_states`
- `5_code/notebooks/jaxley_test_adex.ipynb` — remove geometry hack, use clamp
- `5_code/ADoptEX/core/simulation.py` — update `create_adex_cell()` and simulation wrappers
- `5_code/ADoptEX/training/trainer.py` — update cell setup if it touches geometry/current

### For Approach B
- `5_code/ADoptEX/core/parameters.py` — add `naud_to_density()`, `density_to_naud()`
- `5_code/notebooks/jaxley_test_adex.ipynb` — use conversion at setup, simple I_nA = I_pA/1000
- `5_code/ADoptEX/core/simulation.py` — integrate conversion into simulation wrappers
- Possibly `5_code/ADoptEX/training/trainer.py` — bounds conversion
- `3_jaxley/jaxley/channels/non_capacitive/adex.py` — NO changes needed

### For Both
- Re-validate against Brian2 after changes (notebook cells 26-28)
- Update `CLAUDE.md` and `architecture.html` per documentation sync checklist