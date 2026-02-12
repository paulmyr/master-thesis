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

### 1. Gradient Clipping via `optax.clip_by_global_norm`

**Why it helps:**
Directly caps the gradient norm spikes (10^4 - 10^8) that cause divergence. The most common
first-line defense against exploding gradients.

**What already exists:**
- `optax.clip_by_global_norm(max_norm)` -- one-liner, composable via `optax.chain()`

**What needs to change in ADoptEX:**
- Add `grad_clip_norm: float | None = None` to `TrainingConfig`
- In `_create_optimizer()`, wrap with `optax.chain(optax.clip_by_global_norm(clip), optimizer)`
  when `grad_clip_norm` is set

**Scope:** ~10 lines in `trainer.py` + tests

---

### 2. Learning Rate Schedule

**Why it helps:**
High LR early for fast convergence, low LR later to avoid overshooting good basins. Prevents
the "catapulted out" behavior observed after epoch 19.

**What already exists:**
- `optax.cosine_decay_schedule(init_value, decay_steps)`
- `optax.exponential_decay(init_value, transition_steps, decay_rate)`
- `optax.warmup_cosine_decay_schedule(...)` for warmup + decay

**What needs to change in ADoptEX:**
- Add `lr_schedule: Literal["constant", "cosine", "exponential"] | None = None` and
  `lr_schedule_kwargs: dict | None = None` to `TrainingConfig`
- In `_create_optimizer()`, construct a schedule and pass it as the learning rate

**Scope:** ~20 lines in `trainer.py` + tests

---

### 3. Polyak Gradient Normalization

**Why it helps:**
Deistler's primary optimizer for biophysical models. Normalizes the gradient step by
`||grad||^beta`, eliminating gradient magnitude variation entirely. Optionally multiplies by the
loss value so the step size automatically decays as the loss decreases.

Update rule: `params -= lr * loss^alpha * grad / ||grad||^beta`

With `alpha=1, beta=1` this becomes: `params -= lr * loss * grad / ||grad||` (Deistler default).

**What already exists:**
- `jaxley.optimize.utils.l2_norm(pytree)` -- computes `||pytree||_2` over a JAX pytree
- Pattern demonstrated in `3_jaxley/docs/examples/00_l5pc_gradient_descent.ipynb`

**What needs to change in ADoptEX:**
- Add `"polyak"` as an option to `TrainingConfig.optimizer`
- Implement manual gradient normalization in the training loop (this is not a standard optax
  optimizer -- it requires access to the loss value)
- The training step becomes:
  ```python
  loss, grad = jax.value_and_grad(loss_fn)(params)
  norm = l2_norm(grad)
  params = jax.tree.map(lambda p, g: p - lr * loss * g / norm, params, grad)
  ```

**Scope:** ~40 lines in `trainer.py` + tests

---

### 4. Parameter Transformations (Sigmoid Reparameterization)

**Why it helps:**
Replaces hard clipping (`_clip_trainable_params`) with a smooth bijective transformation.
The optimizer works in unconstrained space; parameters are always in bounds after applying
`transform.forward()`. Eliminates gradient discontinuities at bound edges where clipping
currently kills gradients.

**What already exists:**
- `jaxley.optimize.transforms.SigmoidTransform(lower, upper)` -- maps R -> [lower, upper]
- `jaxley.optimize.transforms.ParamTransform` -- applies per-parameter transforms to pytrees
- Both have `.forward()` (unconstrained -> constrained) and `.inverse()` (constrained -> unconstrained)

**What needs to change in ADoptEX:**
- Add `use_param_transform: bool = False` to `TrainingConfig`
- In `train()`:
  1. Build a `ParamTransform` from `PARAM_BOUNDS` using `SigmoidTransform(bound.min, bound.max)`
  2. Apply `transform.inverse()` to initial params (constrained -> unconstrained)
  3. Optimize in unconstrained space
  4. Apply `transform.forward()` when evaluating the loss
  5. Apply `transform.forward()` to final params before returning
- Remove hard clipping when transforms are active

**Scope:** ~30 lines in `trainer.py` + update `setup_trainable_cell()` + tests

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

Suggested batches:
1. **Batch A** (immediate): Items 1 + 2 -- optax-only changes, minimal risk
2. **Batch B** (next): Items 3 + 4 -- training loop changes, moderate complexity
3. **Batch C** (then): Items 5 + 6 -- loss preprocessing + per-param LR
4. **Batch D** (if needed): Items 7 + 8 -- multi-start and TBPTT
