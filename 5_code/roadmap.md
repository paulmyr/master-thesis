# Training Stabilization Roadmap

## Problem

Training is unstable: gradient norms spike from ~10 to 10^8, loss is non-monotonic, and the
optimizer gets catapulted out of good basins (best loss at epoch 19, then diverges through epoch 75).

Deistler et al. 2025 describe 6 techniques for robust training of biophysical models in Jaxley.
Most are already available as building blocks in the Jaxley fork at `3_jaxley/`. The ADoptEX
trainer (`training/trainer.py`) currently uses none of them.

## Items

Ordered by effort/impact ratio (lowest hanging fruit first).

---

### 1. Gradient Clipping via `optax.clip_by_global_norm` -- DONE

**Why it helps:**
Directly caps the gradient norm spikes (10^4 - 10^8) that cause divergence. The most common
first-line defense against exploding gradients.

**What already exists:**
- `optax.clip_by_global_norm(max_norm)` -- one-liner, composable via `optax.chain()`

**Implemented:**
- Added `grad_clip_norm: float | None = None` to `TrainingConfig` with `__post_init__` validation
- `_create_optimizer()` prepends `optax.clip_by_global_norm()` via `optax.chain()` when set
- `_compute_grad_norm()` still reports raw (pre-clip) norms for diagnostics
- 28 new tests in `tests/test_training/test_trainer.py`

---

### 2. Learning Rate Schedule -- DONE

**Why it helps:**
High LR early for fast convergence, low LR later to avoid overshooting good basins. Prevents
the "catapulted out" behavior observed after epoch 19.

**What already exists:**
- `optax.cosine_decay_schedule(init_value, decay_steps)`
- `optax.exponential_decay(init_value, transition_steps, decay_rate)`
- `optax.warmup_cosine_decay_schedule(...)` for warmup + decay

**Implemented:**
- Added `lr_schedule: Literal["constant", "cosine", "exponential"] | None = None` and
  `lr_decay_rate: float = 0.01` to `TrainingConfig` with `__post_init__` validation
- `_create_optimizer()` passes schedule callable as `learning_rate` to base optimizer
- `lr_history: list[float]` added to `TrainingResult` (populated when schedule active)
- Enhanced logging: per-epoch LR in periodic log, LR decay range in summary
- Tests cover convergence, LR history population, and SGD compatibility

---

### 3. Polyak Gradient Normalization -- DONE

**Why it helps:**
Deistler's primary optimizer for biophysical models. Normalizes the gradient step by
`||grad||^beta`, eliminating gradient magnitude variation entirely. Optionally multiplies by the
loss value so the step size automatically decays as the loss decreases.

Update rule: `params -= lr * loss^alpha * grad / ||grad||^beta`

With `alpha=1, beta=1` this becomes: `params -= lr * loss * grad / ||grad||` (Deistler default).

**What already exists:**
- `jaxley.optimize.utils.l2_norm(pytree)` -- computes `||pytree||_2` over a JAX pytree (not
  exported from `jaxley.optimize`; only a bare helper). Jaxley does NOT implement Polyak as a
  reusable optimizer — the pattern is only shown inline in
  `3_jaxley/docs/examples/00_l5pc_gradient_descent.ipynb`.

**Implemented:**
- Added `"polyak"` to `TrainingConfig.optimizer` Literal type
- Added `polyak_alpha: float = 1.0` (loss exponent) and `polyak_beta: float = 1.0` (grad norm
  exponent) to `TrainingConfig`
- `__post_init__` validation: `polyak_beta` must be positive; `lr_schedule` and `grad_clip_norm`
  are incompatible with polyak (raises `ValueError`)
- `_create_optimizer()` uses `optax.inject_hyperparams(optax.sgd)` for polyak to allow per-step
  LR mutation
- Training loop: normalizes grads by `||grad||^beta + 1e-8`, sets effective LR to
  `lr * loss^alpha` each step
- `lr_history` tracks effective LR (decays automatically as loss decreases)
- `clipped_grad_norms` stays empty (normalization replaces clipping)
- Reuses existing `_compute_grad_norm()` (no new Jaxley dependency)
- 8 new tests in `tests/test_training/test_trainer.py` (convergence, LR tracking, alpha/beta
  edge cases, incompatibility checks, defaults)

---

### 4. Parameter Transformations (Sigmoid Reparameterization) -- DONE

**Why it helps:**
Replaces hard clipping (`_clip_trainable_params`) with a smooth bijective transformation.
The optimizer works in unconstrained space; parameters are always in bounds after applying
`transform.forward()`. Eliminates gradient discontinuities at bound edges where clipping
currently kills gradients.

**What already exists:**
- `jaxley.optimize.transforms.SigmoidTransform(lower, upper)` -- maps R -> [lower, upper]
- `jaxley.optimize.transforms.ParamTransform` -- applies per-parameter transforms to pytrees
- Both have `.forward()` (unconstrained -> constrained) and `.inverse()` (constrained -> unconstrained)

**Implemented:**
- Added `use_param_transform: bool = False` to `TrainingConfig` (default off, backward compatible)
- `__post_init__` warns when `use_param_transform=True` and `clip_to_bounds=True` (transform
  implicitly enforces bounds; hard clipping is skipped)
- New private helpers in `trainer.py`:
  - `_param_key_to_bounds_key()` — extracts `AdEx_` prefix stripping + `capacitance` → `C_m`
    mapping (previously duplicated in 3 places)
  - `_build_param_transform()` — builds `ParamTransform` from `PARAM_BOUNDS` using
    `SigmoidTransform(lower, upper)` for each trainable parameter. Lazy import of Jaxley
    transforms.
  - `_nudge_from_bounds()` — nudges params at exact bounds inward by `eps = 1e-4 * range`
    to prevent `sigmoid.inverse()` returning +/-Inf
- In `train()`:
  1. Builds transform, nudges params, applies `inverse()` to enter unconstrained space
  2. Wraps `loss_fn` with `transform.forward()` so loss sees constrained values
  3. Skips hard clipping in training loop when transform is active
  4. Applies `transform.forward()` for periodic logging (human-readable values)
  5. Applies `transform.forward()` to output params before returning `TrainingResult`
- Compatible with all optimizers (Adam, SGD, RMSProp, Polyak) and `return_best`
- 20 new tests in `tests/test_training/test_trainer.py` (helper unit tests + integration)

---

### 5. Soft-DTW Preprocessing (Deistler Recipe)

**Why it helps:**
Current DTW loss in `loss/dtw.py` operates on raw voltage traces. Deistler et al. use
preprocessing that dramatically improves trace alignment:
1. Sliding window maximum (smooths spikes into peaks)
2. Rescaling to unit interval [0, 1]
3. Temporal penalty in cost matrix: `c(x_i, y_j) = |x_i - y_j| + lambda * |i - j|`

The temporal penalty discourages warping paths that deviate too far from the diagonal, preventing
pathological alignments where one spike absorbs multiple target spikes.

**What already exists:**
- Basic soft-DTW implementation in `loss/dtw.py` (softmin, cost matrix, DTW alignment)

**What needs to change in ADoptEX:**
- Add `sliding_window_max(trace, window_size)` preprocessing function
- Add `rescale_to_unit_interval(trace)` preprocessing function
- Add `temporal_penalty: float | None = None` to `SoftDTWLossConfig`
- Modify cost matrix construction: `C[i,j] = |x_i - y_j| + lambda * |i - j|`
- Apply preprocessing in `make_soft_dtw_loss_fn()` before DTW computation

**Scope:** ~50 lines in `dtw.py` + tests

---

### 6. Per-Parameter-Group Optimization

**Why it helps:**
Different AdEx parameters have vastly different scales and gradient magnitudes:
- Conductances: ~1-100 nS
- Time constants: ~1-1000 ms
- Voltages: ~-80 to +20 mV
- Adaptation: ~0-100 pA

A single learning rate means one parameter type dominates optimization. Per-group LRs let each
parameter type converge at its natural rate.

**What already exists:**
- `jaxley.optimize.optimizer.TypeOptimizer` -- wraps any optax optimizer, applies different
  `optimizer(learning_rate=...)` per parameter name

**What needs to change in ADoptEX:**
- Add `per_param_lr: dict[str, float] | None = None` to `TrainingConfig`
- In `_create_optimizer()`, construct a `TypeOptimizer` when `per_param_lr` is set
- The `TypeOptimizer` maps parameter names to separate optimizer instances

**Scope:** ~30 lines in `trainer.py` + tests

---

### 7. Multi-Start Optimization + Selection

**Why it helps:**
The AdEx parameter space is highly non-convex with many local minima. Running N parallel
optimizations from different starting points and selecting the best hedges against bad
initializations. Deistler et al. use 1000 parallel runs.

**What already exists:**
- `jax.vmap` for batched computation across parameter initializations
- `jax.random.split` for generating multiple random keys

**What needs to change in ADoptEX:**
- New function `multi_start_train(loss_fn, initial_params_batch, config) -> best_result`
- Uses `jax.vmap` over the loss function for parallel evaluation
- Runs N independent gradient descent loops (can be vmapped if loop is pure JAX)
- Selects best result by lowest final loss or best coincidence factor
- Notebook-level orchestration for parameter initialization sampling

**Scope:** ~60 lines, new function in `trainer.py` + notebook-level usage

---

### 8. Truncated Backpropagation Through Time

**Why it helps:**
Limits gradient propagation to windows of N timesteps, preventing gradient explosion over long
simulation sequences. Instead of backpropagating through the entire 2000+ timestep simulation,
gradients are computed over shorter chunks.

**What already exists:**
- `jaxley.utils.jax_utils.nested_checkpoint_scan(f, init, xs, ...)` -- memory-efficient scan
  with gradient checkpointing, supports nested structure for truncated BPTT

**What needs to change in ADoptEX:**
- Integration into simulation wrapper in `core/simulation.py`
- Requires restructuring how `jx.integrate()` is called -- currently treated as a black box
- May need custom integration loop that uses `nested_checkpoint_scan` instead of `jx.integrate`

**Scope:** Significant refactor of simulation layer. Lowest priority unless memory is a
bottleneck or gradient explosion persists after items 1-4.

---

### 9. Make C_m (Membrane Capacitance) Trainable -- DONE

**Why it helps:**
The optimizer currently trains 7 AdEx parameters but excludes C_m (membrane capacitance). C_m
controls the membrane time constant τ_m = C_m / g_L, which determines how quickly the neuron
responds to input. Fitting τ_m is likely important for matching spike timing and subthreshold
dynamics, but the current architecture makes C_m difficult to train due to a dual-capacitance
problem.

**The dual-capacitance problem:**
Two separate C_m values exist in the system:
1. **Channel parameter** `AdEx_C_m` — stored in AdExSurrogate's channel params, used in
   `update_states()` to compute voltage updates via forward Euler
2. **Compartment parameter** `capacitance` — Jaxley's built-in compartment property, used by
   the solver to apply external current: `v += dt * i_ext / capacitance`

These must stay in sync. If only `AdEx_C_m` is trained, the external current contribution uses a
stale `capacitance` value, producing physically inconsistent dynamics. If both are made trainable
independently, they can diverge.

**Current voltage update architecture:**
The membrane potential is updated in two steps each timestep:
1. **Channel `update_states()`**: Forward Euler step using `AdEx_C_m`:
   `v += dt/C_m * (-g_L*(v-E_L) + g_L*ΔT*exp(...) - w)`
2. **Solver external current**: Jaxley adds `dt * i_ext / capacitance` after channel updates

Note: `simulation.py` pre-scales the current by `1/C_m` before passing to Jaxley, making the
external current step effectively `dt * (i_ext/C_m) / capacitance`. If `capacitance=1` (Jaxley
default for single-compartment), the pre-scaling makes the external current C_m-independent at
the compartment level — but this breaks if `capacitance` is changed.

**Approach:**
1. Remove `AdEx_C_m` from AdExSurrogate channel parameters entirely
2. Have `update_states()` read `params["capacitance"]` (Jaxley's compartment parameter) instead
3. Remove the `1/C_m` pre-scaling in `simulation.py` — let the solver handle `i_ext / capacitance`
   naturally
4. Make `capacitance` trainable via Jaxley's `cell.make_trainable("capacitance")` in `trainer.py`
5. Add `capacitance` to `PARAM_BOUNDS` with appropriate bounds (e.g., 50-500 pF)

**Files to change:**
- `3_jaxley/.../adex.py` — remove `AdEx_C_m` param, read `params["capacitance"]` in
  `update_states()`
- `5_code/ADoptEX/core/simulation.py` — remove current pre-scaling by `1/C_m`
- `5_code/ADoptEX/training/trainer.py` — add `capacitance` to trainable parameters
- `5_code/ADoptEX/core/parameters.py` — add `capacitance` to `PARAM_BOUNDS`

**Verification:**
- Re-validate AdExSurrogate against Brian2 reference (<1ms spike timing difference)
- Confirm gradients flow through `capacitance` (non-zero ∂loss/∂C_m)
- Test that trained C_m stays within physiological bounds

**Scope:** Moderate (~50 lines across 4 files). Requires careful testing since it changes the
core simulation dynamics.

**Implementation notes (DONE):**
- Removed `AdEx_C_m` from channel_params; both `AdEx.update_states()` and
  `AdExSurrogate.update_states()` now read `params["capacitance"]` (Jaxley passes this
  to channels via `_step_channels_state()`)
- Kept current pre-scaling (`I_nA = I_pA * C_m / 1000`) — unit analysis shows `C_m_init`
  in the numerator cancels with the `C_m_init`-derived area in the denominator
- When user includes `"C_m"` in `trainable_params`, it maps to `cell.make_trainable("capacitance")`
- Added `"capacitance"` alias in `PARAM_BOUNDS` with same bounds as `"C_m"`
- All `capacitance` → `C_m` mappings added in `_clip_trainable_params`, `_format_params`,
  `get_params_dict`, and `convert_trainable_to_params`
- C_m is NOT in the default trainable params list; users opt in explicitly
- 200 tests pass (7 new tests added)

---

## Diagnosis: Why Items 1-3 Did Not Improve Training

Items 1-3 (gradient clipping, LR schedules, Polyak normalization) address **gradient magnitude**
problems but not the **gradient direction** or **gradient existence** problems. Analysis of the
AdExSurrogate implementation and training dynamics reveals that gradients are not just unstable —
they are often zero or pointing in unhelpful directions.

### Root Cause 1: Surrogate Gradient Too Narrow

The default `surrogate_slope=25.0` creates an effective gradient window of only ~0.4 mV around
the threshold. For a neuron whose membrane potential ranges from -70 to +35 mV, this means
gradients are non-zero for only 0.4% of the voltage range. Any timestep where v is more than
0.2 mV away from threshold contributes zero gradient.

**Evidence:** Gygax & Zenke 2025 show theoretically that sharper surrogates (higher β) produce
more accurate gradient estimates but cause more "dead neurons" — neurons stuck in a regime where
no surrogate gradient flows. They recommend β ≈ 5-10 for training stability, much lower than our
β = 25.

**Fix:** Lower `surrogate_slope` to 5-10. This widens the gradient window to ~2-4 mV, allowing
gradient signal from more timesteps while still providing a reasonable approximation.

### Root Cause 2: v_threshold Creates a 65 mV Gradient Dead Zone

The AdExSurrogate uses `v_threshold=35.0` mV for spike detection, but the exponential term in the
AdEx equation is clamped: `exp_arg = jnp.clip(exp_arg, a_max=10.0)` (line 290 in
`channels/non_capacitive/adex.py`). This clamp activates when `(v - v_thresh) / delta_T > 10`,
which for typical `delta_T ≈ 2 mV` means clamping at `v ≈ v_thresh + 20 ≈ -10 mV`.

The surrogate gradient is centered at `v_threshold=35.0`, but the exponential dynamics already
saturate at v ≈ -10 mV. Between -10 mV and +35 mV there is a 45 mV zone where the forward pass
is clamped (zero gradient through exp) AND the surrogate is inactive (v still below threshold).
This creates a massive gradient dead zone.

**Fix:** Lower `v_threshold` toward 0.0 mV, closer to where the exp clamp kicks in, so the
surrogate gradient activates in a region where the forward dynamics still have gradient flow.

### Root Cause 3: Vanishing Gradients Through Time

The simulation runs for ~5000 timesteps. Gradients must backpropagate through the entire sequence.
The leak dynamics apply a multiplicative factor < 1 at each timestep:
`v_new = v + dt/C * (-g_L * (v - E_L) + ...)`. The gradient through this recurrence decays
exponentially with the number of timesteps, making it nearly impossible for early-trace events
to influence parameter gradients.

**Evidence:** This is the standard vanishing gradient problem in RNNs, but worse because the
biophysical leak term is an inherent dampening mechanism. Zenke & Ganguli 2018 (SuperSpike) and
Bellec et al. 2020 (e-prop) both identify this as a fundamental challenge.

**Impact:** Even if the surrogate gradient is non-zero at the threshold crossing, the gradient
signal decays to near-zero by the time it reaches the parameters. This makes items 1-3 ineffective:
you can't normalize or clip a gradient that is already effectively zero.

### Root Cause 4: Hard Parameter Clipping Kills Gradients at Bounds

`_clip_trainable_params()` applies `jnp.clip()` after each gradient step. When a parameter is at
its bound, the gradient points outside the feasible region, and clipping projects it back — but
the gradient information is lost. The optimizer accumulates momentum in a direction that gets
clipped away every step, wasting update capacity.

**Fix:** Sigmoid reparameterization (roadmap item 4). Map parameters through a smooth sigmoid
bijection so the optimizer works in unconstrained space. Parameters are always in bounds and
gradients flow smoothly near the boundaries.

### Root Cause 5: Gradient Clipping/LR/Polyak Can't Fix Gradient Direction

All three implemented techniques only modify gradient **magnitude** or **step size**:
- Gradient clipping caps the norm but preserves direction
- LR schedules reduce step size over time
- Polyak normalizes to unit direction and scales by loss

If the gradient direction itself is wrong (pointing toward non-spiking solutions) or zero (dead
neuron problem), none of these help. The optimizer takes perfectly-sized steps in the wrong
direction.

### Root Cause 6: Loss Function Design Gaps

**MSE loss** penalizes voltage misalignment at every timestep. A slight temporal shift in spike
timing creates a huge positive-negative artifact, pushing the optimizer toward flat (non-spiking)
solutions where MSE is lower.

**Guarino loss** extracts scalar features (spike count, spike times, ISIs), which is better but
still has issues: (1) soft feature extraction with temperature=0.3 may introduce bias, (2) the
relative error formulation can have numerical issues when target features are near zero, and (3)
it doesn't provide temporal credit assignment — telling the optimizer *when* to modify the voltage
trace to get a spike.

**Soft-DTW loss** on raw voltage is dominated by subthreshold dynamics (the voltage is subthreshold
for most of the trace), diluting the spike alignment signal.

### Recommended Interventions (Priority Order)

Based on the diagnosis, the following interventions target the actual root causes:

1. **Lower surrogate_slope to 5-10** (Root Cause 1) — Immediate, zero-code change (notebook
   hyperparameter). Widens gradient window from 0.4 mV to 2-4 mV.

2. **Lower v_threshold to ~0 mV** (Root Cause 2) — Notebook hyperparameter change. Aligns
   surrogate gradient activation with the region where exponential dynamics have gradient flow.

3. **Sigmoid reparameterization** (Root Cause 4, roadmap item 4) — Eliminates gradient death at
   bounds. Already planned, now higher priority.

4. **Add spike count regularizer** (Root Cause 6) — A simple `(n_spikes_sim - n_spikes_target)^2`
   term added to any loss function. Prevents convergence to non-spiking solutions by explicitly
   penalizing wrong spike count. Can use the existing `soft_spike_count()` from guarino.py.

5. **Van Rossum distance** (Root Cause 6) — **IMPLEMENTED** in `ADoptEX/loss/van_rossum.py`.
   Convolves spike trains with exponential kernel `κ(t) = exp(-t/τ)`, then L2 distance.
   Naturally differentiable through `jnp.convolve`, provides temporal credit assignment,
   and is a theoretically grounded spike train metric (van Rossum 2001). The time constant τ
   controls the precision/rate tradeoff. `VanRossumLossConfig` with optional subthreshold
   voltage MAE. 23 tests in `test_loss/test_van_rossum.py`.

6. **Soft-DTW preprocessing** (Root Cause 6, roadmap item 5) — Sliding window max + rescaling
   makes DTW focus on spike peaks rather than subthreshold dynamics. Already planned.

### Literature References for Diagnosis

- **Gygax & Zenke 2025** — "Elucidating the theoretical underpinnings of surrogate gradient
  learning in SNNs". Proves sharper surrogates are more accurate but cause more dead neurons.
  Recommends lower β for stability. Shows surrogate gradient methods are biased estimators of
  the true loss gradient.

- **Zenke & Ganguli 2018** (SuperSpike) — Demonstrates vanishing gradient problem in spiking
  networks, proposes SuperSpike surrogate `1/(β|v-θ|+1)²` with heavy tails for better gradient
  flow.

- **Bellec et al. 2020** (e-prop) — Eligibility propagation as biologically plausible alternative
  to BPTT, addressing vanishing gradients in recurrent spiking networks.

- **Klos et al. 2024** — Reports "pseudospikes" (false spike detections in the surrogate backward
  pass) as a major failure mode. Argues for careful surrogate calibration.

- **Deistler et al. 2025** — Uses MAE on summary statistics (not MSE on raw traces) + Polyak +
  sigmoid transforms + soft-DTW with preprocessing as a complete recipe for biophysical model
  fitting.

- **van Rossum 2001** — Spike train distance metric based on exponential kernel convolution.
  Time constant τ interpolates between rate coding (large τ) and temporal coding (small τ).

---

## Jaxley Building Blocks Reference

| Tool | Location | Signature |
|------|----------|-----------|
| `l2_norm` | `jaxley.optimize.utils` | `l2_norm(x: PyTree) -> Array` |
| `SigmoidTransform` | `jaxley.optimize.transforms` | `SigmoidTransform(lower, upper)` |
| `ParamTransform` | `jaxley.optimize.transforms` | `ParamTransform(lowers, uppers, transforms)` |
| `TypeOptimizer` | `jaxley.optimize.optimizer` | `TypeOptimizer(optimizer, optimizer_params)` |
| `nested_checkpoint_scan` | `jaxley.utils.jax_utils` | `nested_checkpoint_scan(f, init, xs, length, ...)` |
| Polyak pattern | `3_jaxley/docs/examples/00_l5pc_gradient_descent.ipynb` | Manual loop |

## Implementation Order

Items 1-2 are independent and can be done in any order (or together).
Item 3 depends on understanding the training loop changes from 1-2.
Item 4 is independent of 1-3 but changes the parameter flow.
Item 5 is independent (loss module only).
Items 6-8 depend on the training infrastructure from 1-4.
Item 9 is independent of 1-8 but should be done after item 4 (sigmoid reparameterization)
so that C_m bounds are handled smoothly from the start.

Suggested batches:
1. **Batch A** (immediate): Items 1 + 2 -- optax-only changes, minimal risk
2. **Batch B** (next): Items 3 + 4 -- training loop changes, moderate complexity
3. **Batch C** (then): Items 5 + 6 + 9 -- loss preprocessing + per-param LR + trainable C_m
4. **Batch D** (if needed): Items 7 + 8 -- multi-start and TBPTT
