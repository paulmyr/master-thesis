 ▐▛███▜▌   Claude Code v2.1.119
▝▜█████▛▘  Opus 4.7 · Claude Pro
  ▘▘ ▝▝    ~/Projects/university/40_thesis/5_code

❯ Can we make a corresponding plot: lower init value -> more failure? Distance  
  to ground truth can be in both directions, lower and higher, right?           
  ⎿  ⧉ Selected 1 lines from 5_code/notebooks/hparam_sweep_analysis.ipynb in PyCharm

  Read 1 file (ctrl+o to expand)
  ⎿  Loaded 5_code/CLAUDE.md

⏺ Two things to untangle here:

  1. The existing plot already encodes both directions. Signed standardized
  coefficients: red (positive) = higher init → more failure, blue (negative) =
  lower init → more failure. So init_E_L with coef -0.78 already says "lower E_L
   → more failure". The legend just doesn't spell out the blue case. Quickest
  fix is updating the caption in cell a3fdd642:

  Red bar = higher init value -> more failure.
  Blue bar = lower init value -> more failure.
  Stars = bootstrap-significance (200 resamples).

  2. Your second point is the more interesting one — a signed regression assumes
   failure is monotonic in the init value. But for params like E_L or v_T, both
  far-below-GT and far-above-GT inits should hurt; the truth is U-shaped, not
  linear. A signed coefficient of -0.78 for E_L could just mean the sweep
  happened to sample more points above GT than below, and the regression is
  fitting the asymmetry rather than a real "lower is worse" effect.

  The clean companion analysis is per-parameter |init − GT| (absolute
  deviation). Two ways:

  - Cheap: add a cell that computes abs_dev_<param> = |init_<param> − 
  gt_<param>| for each of the 6 params, runs the same logistic regression on
  those 6 features, and plots coefficients. All bars should be red if the
  symmetric story holds; any param with |dev| not predictive is either
  irrelevant or only matters in one direction.
  - More informative: scatter init_<param> − gt_<param> (signed) vs. failure
  rate per param, with GT marked at 0. Reveals whether the U is symmetric,
  one-sided, or actually monotonic.

  Want me to add a new cell after §0c that does the absolute-deviation
  regression + a 2×3 scatter grid of signed deviation vs failure? I'd need to
  either read GT from DEFAULT_PARAMS/NAUD_PARAMETERS or have the sweep CSV log
  it — do you know which one the sweep uses as ground truth?

✻ Cogitated for 35s

❯ yes please                                                                    

  Searched for 1 pattern, read 1 file (ctrl+o to expand)

⏺ Edit Notebook(notebooks/hparam_sweep_analysis.ipynb@a3fdd642)
  ⎿  Updated cell 91840bf7:
         1 ## 0d. Symmetric distance-from-GT analysis
         2 
         3 §0c uses raw init values, so a signed coefficient assumes fail
           ure is monotonic in
         4 the parameter. But for params like `E_L` or `v_T` both far-bel
           ow-GT and far-above-GT
         5 inits should hurt — the failure curve is U-shaped, not linear,
            and the regression
         6 above can confuse "lower → worse" with "the sweep happened to 
           sample asymmetrically".
         7 
         8 Two views here:
         9 - **Absolute deviation** `|init - gt|` per param + same logist
           ic regression. All
        10   significant bars should now be red (more deviation = more fa
           ilure) regardless of direction.
        11 - **Signed scatter** of `init - gt` vs failure rate per param,
            with GT marked at 0.
        12   Reveals whether the U is symmetric, one-sided, or genuinely 
           monotonic.
        13 
        14 Ground truth taken from `NAUD_PARAMETERS["tonic"]` (matches `h
           param_sweep.py`).

⏺ Edit Notebook(notebooks/hparam_sweep_analysis.ipynb@91840bf7)
  ⎿  Updated cell dce57a95:
         1 from ADoptEX.core.parameters import NAUD_PARAMETERS
         2 
         3 gt = dict(NAUD_PARAMETERS["tonic"])
         4 param_names = [c[len("init_"):] for c in init_features]
         5 gt_values = {p: gt[p] for p in param_names}
         6 print('GT (tonic):', {p: round(v, 4) for p, v in gt_values.ite
           ms()})
         7 
         8 # --- Absolute deviation features ---
         9 abs_features = [f"absdev_{p}" for p in param_names]
        10 X_abs_raw = pd.DataFrame({
        11     f"absdev_{p}": (df[f"init_{p}"] - gt_values[p]).abs()
        12     for p in param_names
        13 })
        14 X_abs = StandardScaler().fit_transform(X_abs_raw)
        15 
        16 abs_results = {}
        17 for tgt in targets:
        18     y = df[tgt].astype(int).values
        19     if y.sum() == 0 or y.sum() == len(y):
        20         abs_results[tgt] = None
        21         continue
        22     lr_ = LogisticRegression(max_iter=2000, C=1.0).fit(X_abs, 
           y)
        23     coefs = pd.Series(lr_.coef_[0], index=abs_features)
        24 
        25     rng_ = np.random.default_rng(0)
        26     boot_coefs = np.zeros((200, len(abs_features)))
        27     for i in range(200):
        28         idx = rng_.integers(0, len(y), len(y))
        29         try:
        30             lr_b = LogisticRegression(max_iter=2000, C=1.0).fi
           t(X_abs[idx], y[idx])
        31             boot_coefs[i] = lr_b.coef_[0]
        32         except Exception:
        33             boot_coefs[i] = np.nan
        34     se = np.nanstd(boot_coefs, axis=0)
        35     z = coefs.values / np.where(se > 0, se, np.nan)
        36     pvals = pd.Series(2 * (1 - sstats.norm.cdf(np.abs(z))), in
           dex=abs_features)
        37 
        38     abs_results[tgt] = {'coef': coefs, 'pval': pvals,
        39                         'n_pos': int(y.sum()), 'auc_train': lr
           _.score(X_abs, y)}
        40 
        41 for tgt in targets:
        42     r = abs_results[tgt]
        43     if r is None:
        44         print(f'\n{tgt}: skipped (degenerate)'); continue
        45     print(f'\n=== {tgt} | absolute deviation (n_failures={r["n
           _pos"]}/{len(df)}) ===')
        46     summary = pd.DataFrame({'std_coef': r['coef'], 'pval': r['
           pval']}) \
        47         .sort_values('std_coef', key=lambda s: s.abs(), ascend
           ing=False)
        48     summary['sig'] = summary['pval'].apply(lambda p: '***' if 
           p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '')
        49     print(summary.round(3).to_string())

⏺ Edit Notebook(notebooks/hparam_sweep_analysis.ipynb@dce57a95)
  ⎿  Updated cell 882367c7:
         1 # Coefficient bar plot (absolute deviation)
         2 valid = [t for t in targets if abs_results[t] is not None]
         3 fig, axes = plt.subplots(1, len(valid), figsize=(5 * len(valid
           ), 5), sharey=True)
         4 if len(valid) == 1: axes = [axes]
         5 for ax, tgt in zip(axes, valid):
         6     r = abs_results[tgt]
         7     s = r['coef'].sort_values()
         8     colors = ['red' if c > 0 else 'steelblue' for c in s.value
           s]
         9     ax.barh(s.index, s.values, color=colors)
        10     ax.axvline(0, color='k', linewidth=0.8)
        11     ax.set_xlabel('standardized log-odds coefficient')
        12     ax.set_title(f'{tgt} | |init - GT|\n(n_pos={r["n_pos"]}, t
           rain acc={r["auc_train"]:.2f})')
        13     for i, (name, val) in enumerate(s.items()):
        14         p = r['pval'][name]
        15         sig = '***' if p < 0.001 else '**' if p < 0.01 else '*
           ' if p < 0.05 else ''
        16         if sig:
        17             ax.text(val, i, f' {sig}', va='center', fontsize=9
           )
        18 plt.tight_layout(); plt.show()
        19 print('\nRed bar = larger |init - GT| -> more failure (symmetr
           ic distance, direction-agnostic).')

⏺ Edit Notebook(notebooks/hparam_sweep_analysis.ipynb@882367c7)
  ⎿  Updated cell b10c2494:
         1 # Signed-deviation scatter: failure rate binned along (init - 
           GT) for each param.
         2 # Reveals whether failure is U-shaped (symmetric), one-sided, 
           or monotonic.
         3 fig, axes = plt.subplots(2, 3, figsize=(15, 8))
         4 n_bins = 12
         5 for ax, p in zip(axes.flat, param_names):
         6     dev = df[f"init_{p}"] - gt_values[p]
         7     bins = np.linspace(dev.min(), dev.max(), n_bins + 1)
         8     centers = 0.5 * (bins[:-1] + bins[1:])
         9     binned = pd.cut(dev, bins, include_lowest=True)
        10     fail_rate = df.groupby(binned, observed=True)['any_failure
           '].mean()
        11     counts = df.groupby(binned, observed=True).size()
        12     # align to bin centers
        13     rates = np.array([fail_rate.get(b, np.nan) for b in fail_r
           ate.index])
        14     cs = np.array([counts.get(b, 0) for b in fail_rate.index])
        15     bin_centers = np.array([iv.mid for iv in fail_rate.index])
        16 
        17     ax.bar(bin_centers, rates, width=(bins[1]-bins[0])*0.9,
        18            color=['lightcoral' if c < 20 else 'firebrick' for 
           c in cs],
        19            edgecolor='k', linewidth=0.3)
        20     ax.axvline(0, color='green', linestyle='--', linewidth=1.5
           , label='GT')
        21     ax.axhline(df.any_failure.mean(), color='gray', linestyle=
           ':', label=f'overall ({df.any_failure.mean():.0%})')
        22     ax.set_xlabel(f'init_{p} - GT  (GT={gt_values[p]:.3g})')
        23     ax.set_ylabel('failure rate')
        24     ax.set_title(p)
        25     ax.set_ylim(0, 1)
        26 axes[0, 0].legend(fontsize=8, loc='lower right')
        27 plt.suptitle('Failure rate vs signed deviation from GT (light 
           bars = sparse bins, n<20)', y=1.02)
        28 plt.tight_layout(); plt.show()

⏺ Added §0d with three pieces after §0c:

  1. Markdown explaining why signed regression can mislead and what the new
  views show.
  2. Code cell computing |init_<param> − gt_<param>| (GT from
  NAUD_PARAMETERS["tonic"], matching hparam_sweep.py:38), running the same
  bootstrapped logistic regression on absolute deviations, printing coefficients
   + p-values.
  3. Bar plot of those coefficients (red = more |dev| → more failure).
  4. 2×3 binned-failure-rate plot of signed deviation (init − GT) per parameter,
   with GT marked at 0 and the overall failure rate as a baseline. Light bars
  flag bins with n<20 to avoid over-reading sparse tails.

  Run cells §0d in order — the absolute-deviation regression should turn the
  asymmetric blue/red mix from §0c into a clearer "which params matter at all"
  picture, and the scatter shows whether each param's failure curve is U-shaped,
   one-sided, or monotonic.

✻ Worked for 1m 17s

❯ Given an experimental voltage trace, do you think it is feasable to predict   
  or calculate sensible starting paramters? Is defining general starting        
  parameters for different types of neurons scientific?                         

⏺ Both are reasonable and have precedent in the literature — the trick is being
  honest about what's actually identifiable from the trace.

  What you can extract directly from a trace (cheap, well-established):
  - E_L ≈ resting voltage in the pre-stim window (median of subthreshold
  samples).
  - v_T ≈ voltage at spike onset, measured as the point where dV/dt crosses
  ~10–20 mV/ms on the upstroke (Naud & Gerstner do this).
  - v_reset ≈ voltage minimum within ~1–2 ms after each spike peak.
  - g_L and C_m from a small subthreshold step: τ = C/g_L from the exponential
  decay, and steady-state ΔV/I gives 1/g_L. (Standard membrane fit; works
  whenever you have a hyperpolarizing or weak depolarizing pulse without
  spikes.)
  - ΔT is the hard one — it's the slope-factor of the exponential nonlinearity
  and isn't directly readable. People typically fix it to a literature value (~2
   mV) or fit it from the dV/dt-vs-V phase plane near threshold.
  - Adaptation params (a, b, τ_w) are basically not identifiable from a single
  trace — you need the spike-rate adaptation envelope across the train, ideally
  multiple current steps.

  Is "starting parameters per neuron type" scientific? Yes, and it's already
  standard practice. Naud et al. 2008 and the Allen Institute Cell Types
  database both publish typed parameter sets (your sweep already uses
  NAUD_PARAMETERS["tonic"]). The honest framing is "informed prior" rather than
  "ground truth": you're saying the basin you start the optimizer in matters, so
   start it in the basin that morphology/firing-pattern classification suggests.
   As long as you (a) report the priors you used, (b) show optimization isn't
  doing all the work or all the failing, and (c) ideally test sensitivity by
  re-running from perturbed inits — which is exactly what your hparam sweep does
   — it's defensible.

  For your thesis specifically: §0d is going to tell you which params the
  optimizer is sensitive to. The natural follow-up is a two-stage init — direct
  extraction for the cheap params (E_L, v_T, v_reset, maybe g_L/C_m), typed
  prior for the rest (ΔT, a, b, τ_w). That converts "random uniform over bounds"
   (which is what's killing 60% of trials) into something a reviewer would
  actually call a method.

  Caveat worth flagging: if direct extraction works too well, the optimizer
  story shrinks — you're then mostly demoing that gradient descent doesn't break
   what was already a good init. That's still publishable, but the narrative
  shifts from "gradients fit AdEx" to "gradients refine a heuristic init", which
   is a different (and arguably more honest) claim.

❯ Given the experiments we've done here, and given the state of                 
  @2_thesis/main.pdf (and the fact that a working draft of the thesis should be 
   done within 5 days), recommend a detailed step by step guide on how to       
  proceed.                                                                      
  ⎿  Referenced PDF 2_thesis/main.pdf (98 pages)

  Read 1 file (ctrl+o to expand)

⏺ Here's the honest read: you have a 98-page document with a near-complete
  experimental story (5 loss functions tested, six failure modes synthesised,
  Nelder–Mead baseline at Γ=0.872 vs gradient methods at Γ≤0.04) but Chapter 1
  is empty stubs, Chapter 6 is empty, three abstracts are empty, and Chapter 4
  is mixing methods with results. The hparam sweep we've been analysing today
  isn't in the thesis at all yet — and it shouldn't be, given the deadline. The
  thesis story is already complete without it; the sweep belongs in §5.6 or
  Discussion as one paragraph of supporting evidence, not as a new experiment.

  The single most important reframing: stop running experiments. You have a 
  publishable negative result. Every hour spent on §0d-style analysis from now
  on trades against a chapter that is currently zero words.

  5-day plan

  Day 1 — Triage and structural fixes (no new prose yet)

  Goal: turn the document into something where you only have to write, not also
  decide structure.

  1. Methods/Results split (highest-leverage, ~2 hrs). The repeated METHODS vs 
  RESULTS SPLIT itodos in §4.4.2, §4.5, §5.3.4 are the same problem. Rule: §4.x
  contains zero numbers from your experiments. Move every "Γ=...", "g_L: 11.19 →
   11.18", "parameters barely moved" sentence out of Chapter 4 into Chapter 5.
  Delete the duplicated root-cause and experiment-summary itodo blocks in §4.5.
  2. Chapter 4 opening (~30 min). Replace the current opener with the 4-bullet
  factual overview the itodo on p. 25 already drafted for you. Copy-paste,
  polish.
  3. Hyperparameter table (~30 min). Add Van Rossum column (tau_ms); replace
  planned values with what was actually used; add the β_surrogate: 25.0 → 5.0
  footnote.
  4. Delete the soft-DTW chapter stub (the itodo on p. 39 says you already
  deleted it from "Implementation"). Make sure it lives only in §4.4.2 + §5.3.4.
  5. Kill stale itodos as you go — do not leave the red Todo list visible for
  the supervisor draft.

  End of Day 1: chapter 4/5 boundary clean, hyperparameter table real. No new
  content, but the scaffolding holds weight now.

  Day 2 — Chapter 1 (Introduction) and Chapter 6 (Discussion)

  Both chapters are empty stubs and both are short — knock them out together
  while the project is fresh in your head.

  Chapter 1 (~4 pages, half a day). Use the seven existing subsection headings
  as a forcing function:
  - 1.1 Background — 2 paragraphs: simplified neuron models matter for HPC-scale
   simulation; fitting them is the bottleneck.
  - 1.2 Problem — the differentiability problem in one paragraph.
  - 1.3 Purpose — research question verbatim from CLAUDE.md ("Can gradient-based
   parameter optimization produce useful parameter updates for the AdEx
  model?").
  - 1.4 Goals — three subquestions verbatim.
  - 1.5 Method — one paragraph: synthetic-target benchmark, four loss functions,
   Nelder–Mead baseline.
  - 1.6 Limitations — single trace, synthetic GT, no multi-cell.
  - 1.7 Overview — chapter-by-chapter sentence each.

  Chapter 6 (~3-4 pages). You already have the result. Discussion writes itself
  if you frame it as: (a) what worked (implementation, baseline, failure-mode
  taxonomy), (b) why gradients failed (the six failure modes — but as scientific
   interpretation, not re-listing), (c) what this means for the field (don't
  oversell — it's one model, one loss-function family, one trace; cite Klos et
  al. 2024 on surrogate calibration as the natural follow-up), (d) future work
  (better init via direct trace extraction — your last conversation;
  multi-trace; pseudospikes; eligibility propagation).

  End of Day 2: every chapter has prose. No more empty TOC entries.

  Day 3 — Results chapter cleanup + abstract

  1. Fill in the placeholder numbers in §5.3 (the itodos point to specific
  notebooks: training_synthetic_singlestage_summary_stats.ipynb, etc.). Read
  once, write the numbers in, do not re-run.
  2. §5.6 "Why Gradients Fail" — this is your best chapter. Make sure each of
  the six failure modes has: a one-line claim, the evidence (figure/number), a
  citation. This is where the reviewer's eye goes.
  3. Write the English abstract (~250 words). Structure: problem → method → key
  finding (Γ ≤ 0.04 vs Nelder–Mead 0.872) → why it matters. Keywords are already
   filled in.
  4. Translate to Swedish + German abstracts — this is mechanical, do it last
  when you're tired.

  End of Day 3: results chapter has real numbers, abstract done in three
  languages.

  Day 4 — Figures, references, polish

  1. Loss landscape figures (§5.2, §A.1). The four
  loss_landscape_synthetic_all_* notebooks already exist; export the existing
  figures, write the comparison paragraph the itodo on p. 44 asks for ("plateau
  shape differs across losses").
  2. Bibliography sweep. Ten papers in CLAUDE.md; check each is cited and in
  fordiva.json. Add Bellec 2020 and Klos 2024 (the itodo on p. 20 explicitly
  asks).
  3. Front matter: acknowledgements, fix the "THESIS SUB TOPIC" placeholder on
  the cover, fix Swedish title, set keywords.
  4. Strip the visible Todo list before sending to supervisor — they don't need
  to see your scaffolding.

  Day 5 — Read-through + buffer

  1. Read the whole thing front-to-back, once. Mark inconsistencies (section
  number drift, "Figure 5.1" referring to wrong figure, etc.). Do not rewrite —
  just patch.
  2. Buffer day for one of the things that always goes wrong: BibTeX, the
  cleardoublepage itodo on p. 55, broken cross-references, KTH template
  metadata, supervisor wants one more thing.

  What to explicitly NOT do this week

  - Do not add the hparam sweep as a new section. If you mention it at all: one
  paragraph in Discussion under "robustness", citing it as supporting evidence
  for the "non-convex parameter space" failure mode. The §0d analysis we just
  built is for you, not the thesis.
  - Do not implement direct-trace parameter extraction (last conversation's
  idea). It's a great future-work paragraph in Chapter 6.
  - Do not run more experiments. The Γ ≈ 0 vs 0.872 contrast is the result. More
   data won't change the story; missing chapters will sink it.
  - Do not refactor ADoptEX further. The library is fine.

  The one risk to flag now

  Your Chapter 5 introduction (p. vi itodo) admits "all gradient-based
  configurations fail" and "Nelder–Mead consistently outperforms every
  gradient-based variant." A reviewer will ask: is this a result about gradient 
  methods, or about your particular implementation? Your defence is §5.6 — the
  six-failure-mode taxonomy — and the loss-landscape figures (§5.2) showing the
  geometry, not the optimiser, is the problem. Make sure those two sections are
  the strongest in the document. Everything else is supporting cast.

✻ Sautéed for 1m 19s

❯ I have multiple issues here. First, my results are not really valid. All      
  sucessful runs (using gradients) were finley tuned hyperparameter settings    
  and close starting parameters. All non successful runs (gamma = 0 vs 0.872)   
  were run on random hyperparameters with no thought or reasoning behind them.  
  I need a good fundamental experiment plan that I can execute now, where I     
  just have to collect the data.                                                

⏺ You're right, and this is fixable in ~1 day of compute + 1 day of writing. The
   core flaw is unmatched protocols — the cure is one benchmark script that runs
   every method under the same conditions. Critically: pick the design now,
  pre-commit to it, then only collect data. If you start tuning after seeing
  results, you're back where you started.

  The benchmark: one trace, four methods, matched protocol

  Fixed across all methods (write these down, don't change them):
  - Target: the same synthetic NAUD-tonic trace you already use in
  hparam_sweep.py. One trace. No multi-trace generalisation claim.
  - Trainable parameters: same set as the sweep (C_m, g_L, v_reset, v_T, E_L, 
  delta_T). Adaptation params fixed at GT — declare this as a scoping decision
  in §4.3.
  - Init distribution: 50 initial parameter vectors, sampled once from
  PARAM_BOUNDS with a fixed seed. Stratified into 5 distance bins (0.0–0.1,
  0.1–0.2, …, 0.4–0.5 normalised), 10 inits per bin. Every method runs on the 
  same 50 inits. This is the single most important fix — it eliminates init-luck
   as a confound and lets you plot success-rate vs init-distance per method.
  - Compute budget: same wall-clock cap per run for everyone (e.g., 60 s) OR
  same number of objective evaluations (e.g., 500). Pick one and apply
  uniformly. Wall-clock is the more honest choice given gradient methods are
  vmap-friendly and Nelder–Mead isn't.

  Hyperparameters: pre-registered, not tuned on the test set.
  For each gradient method, pick one hyperparameter setting and justify it from
  sources outside this benchmark:
  - surrogate_slope = 5.0 — Gygax & Zenke 2025 recommendation.
  - temperature = 0.3, beta = 7.0, lr = 0.1, polyak_alpha = 1.0, polyak_beta = 
  0.75 — medians of the §0c "successful" trials in your hparam sweep. Cite the
  sweep as a calibration step, not as a result.
  - Adam, no LR schedule, 500 epochs, gradient clipping at 1.0.
  - For Nelder–Mead: scipy defaults, maxiter=500.

  The point: a reviewer can disagree with these choices, but they can't accuse
  you of cherry-picking, because the hyperparameter values are fixed before any
  run on the benchmark and justified by external evidence.

  Methods (5 total):
  1. MSE (gradient)
  2. Guarino (gradient)
  3. Van Rossum (gradient)
  4. Nelder–Mead on Guarino loss (the same loss the gradient method uses —
  apples-to-apples)
  5. Random search (uniform sampling within bounds, same compute budget) — a
  chance-level floor. If gradient + Guarino can't beat random, that's a powerful
   negative result on its own.

  (Drop soft-DTW. Your own itodos say it's incomplete, and trimming a method is
  more honest than padding the table.)

  What to log per run (CSV row)

  method, init_seed, init_dist_bin, init_<param>×6,
  final_<param>×6, gamma, n_spikes_sim, n_spikes_target,
  first_spike_err_ms, min_loss, n_evals, wall_time_s, status

  status ∈ {ok, crashed, timeout}. No reconstruction from seeds — log everything
   (you already have a memory about this).

  Plots and tables (these write the results chapter)

  1. Table 5.1: per-method success rate (Γ > 0.5), median Γ, IQR, median
  spike-count error, median wall time. Five rows.
  2. Figure: Γ violin plot per method, one violin per distance bin. This is your
   money figure — it shows simultaneously that (a) gradients fail across all
  distances, (b) Nelder–Mead degrades with distance but recovers in close bins,
  (c) random is at chance.
  3. Figure: success-rate-vs-init-distance line plot per method.
  4. Figure: a representative trial trace per method at a medium-distance init
  (existing fit_comparison_plot).

  That's it. Three figures + one table. Each maps directly to a Discussion
  claim.

  File layout (start in 30 minutes)

  5_code/
    benchmark.py              # ~150 lines, no argparse
    results_benchmark.csv     # already in your git status
    notebooks/
      benchmark_analysis.ipynb  # plots + table from CSV

  benchmark.py skeleton:

  # Generate 50 stratified inits (one fixed seed)
  # For each method in [mse, guarino, van_rossum, nelder_mead, random]:
  #   For each of the 50 inits:
  #       Run with fixed hparams, fixed budget (60 s)
  #       Log CSV row

  No multiprocessing — your memory file already says JAX + fork deadlocks. Use
  jax.vmap over inits per method if you want speed; otherwise serial is fine for
   250 runs.

  Adjusted 5-day plan

  ┌──────┬──────────────────────────────────────────────────────────────────┐
  │ Day  │                               Goal                               │
  ├──────┼──────────────────────────────────────────────────────────────────┤
  │ Day  │ Write benchmark.py. Test on 2 inits × 5 methods to verify it     │
  │ 1 AM │ runs end-to-end. Commit.                                         │
  ├──────┼──────────────────────────────────────────────────────────────────┤
  │      │ Kick off full run (50 × 5 = 250 runs, ~4–8 h depending on        │
  │ Day  │ budget). Run overnight. Do not touch the script after launch.    │
  │ 1 PM │ While it runs: do the methods/results split from yesterday's     │
  │      │ plan (Day 1 of the original plan).                               │
  ├──────┼──────────────────────────────────────────────────────────────────┤
  │ Day  │ Analysis notebook: table + 3 figures. Write §5.x prose around    │
  │ 2    │ them. Write §5.6 (six failure modes) using the new evidence.     │
  ├──────┼──────────────────────────────────────────────────────────────────┤
  │ Day  │ Chapter 1 + Chapter 6 (as per yesterday's plan).                 │
  │ 3    │                                                                  │
  ├──────┼──────────────────────────────────────────────────────────────────┤
  │ Day  │ Abstracts, bibliography, figures polish, kill stale itodos.      │
  │ 4    │                                                                  │
  ├──────┼──────────────────────────────────────────────────────────────────┤
  │ Day  │ Front-to-back read, buffer for the thing that breaks.            │
  │ 5    │                                                                  │
  └──────┴──────────────────────────────────────────────────────────────────┘

  Two questions to answer before writing any code

  1. Wall-clock or function-eval budget? I'd vote wall-clock, because that's
  what an HPC user actually cares about and Jaxley's whole pitch is GPU speed.
  But you must pick one and justify it in §4.3.
  2. Do you accept dropping soft-DTW from the comparison? If you keep it, you
  need a working implementation today. If you drop it, you save half a day and
  the story tightens. Recommend dropping.

  Both decisions, not the code, are what makes this defensible. Once you've
  answered them, the script is ~2 hours.

✻ Sautéed for 1m 10s
❯ I prefer the function-eval budget. IMO it's unfair to compare jax optimized   
  algorithms against non optimized scipy algorithms in walltime.                
  I accept dropping soft-DTW                                                    

⏺ Both good calls. Function-eval budget actually leads to a cleaner methods
  paragraph because you can defend it from first principles: the AdEx simulation
   is the bottleneck on every method — JAX, scipy, or anything else — so
  counting simulations equalises the workload regardless of optimiser plumbing.

  One detail to lock down: what counts as one eval?

  A backward pass through jax.grad costs ~the same as one forward pass
  (reverse-mode AD on a scalar loss is at most ~3× forward, in practice ~1–2×
  for scan-based simulators). So:

  - Gradient methods: 1 step = 1 forward + 1 backward ≈ 2 sims
  - Nelder–Mead: 1 simplex eval = 1 sim
  - Random search: 1 sample = 1 sim

  Budget = 500 forward-simulation-equivalents for everyone. That gives gradient
  methods 250 Adam steps, Nelder–Mead 500 simplex evals, random search 500
  samples. State this conversion explicitly in §4.3 — it's one sentence and it
  shuts down the obvious reviewer objection.

  If you want to be conservative, count a backward as 2 sims (so 500 budget =
  167 Adam steps). I'd start with the 1:1 ratio because that's what
  jax.value_and_grad actually costs in the scan-heavy regime, and document the
  assumption.

  Finalised setup

  - Methods (4): MSE-grad, Guarino-grad, VanRossum-grad, Nelder–Mead-on-Guarino,
   Random.
  - Wait — that's 5. Stick with 5; random is cheap and the chance-level floor is
   genuinely informative. Drop random only if it complicates the story.
  - Budget: 500 forward-sim-equivalents per run.
  - Inits: 50, stratified into 5 distance bins of 10. One fixed seed for the
  init batch — same 50 vectors hit every method.
  - Hparams: pre-registered (surrogate=5, temp=0.3, β=7, lr=0.1, polyak_α=1,
  polyak_β=0.75 from §0c medians; cite hparam sweep as calibration in §4.3, not
  as result).
  - Stopping: only the budget. No early stopping, no return-best, no
  NaN-recovery — keep the comparison simple. (You can mention return-best as an
  additional technique in §4.5 without running it as a separate condition.)

  Before you start coding — two things to nail in 5 min

  1. Sim length per run. Your hparam sweep used t_max = 100 ms (4000 timesteps
  at dt=0.025). Keep that. Don't change to a longer trace just because the
  benchmark "deserves" it — consistency with the loss-landscape figures matters
  more.
  2. Random search's "answer". It samples 500 points; report the best one
  (lowest loss). Anything else is a strawman.

  Ship benchmark.py this morning, launch by lunch, analyse tomorrow. Want me to
  draft the script skeleton, or do you have it?

❯ Yes, plan out benchmark.py                                                    

⏺ Explore(Map benchmark scaffolding and library APIs)
  ⎿  Done (27 tool uses · 75.8k tokens · 1m 12s)
  (ctrl+o to expand)

  Read 1 file (ctrl+o to expand)

⏺ I have what I need. The existing benchmark.py is 422 lines (well over your
  "50-150 lines" rule) and uses 3 scenarios × 10 uniform seeds × 4 methods with
  mismatched budgets — a partial rewrite is the right call. Let me write the
  plan.

⏺ Updated plan
  ⎿  /plan to preview
  ⎿  Tool use rejected

⏺ User answered Claude's questions:
  ⎿  · How should the 3 scenarios share compute? At 50 inits/bin × 5 bins × 5 
     methods, each scenario is ~1250 runs (~10-20h overnight). Three scenarios 
     full-fat would take 30-60h. → All three at 20 inits/bin
     · For adaptation and initial_bursting scenarios, which AdEx parameters 
     should be trainable? → Keep same 6 params, fix a, b, tau_w at GT
     · How should per-method hyperparameters be specified? → Flat METHOD_CONFIGS
      dict at top of file (Recommended)

⏺ Updated plan
  ⎿  /plan to preview

⏺ User approved Claude's plan
  ⎿  Plan saved to: ~/.claude/plans/yes-plan-out-benchmark-py-snappy-diffie.md ·
      /plan to edit
     benchmark.py — matched-protocol comparison

     Context

     Current results in the thesis are not defensible: gradient "wins" used
     hand-tuned hparams + close inits; gradient "losses" used random hparams.
     The Γ ≈ 0 vs Γ = 0.872 contrast in §5.5 would be rejected by any reviewer.
     We need one benchmark, run with matched protocol, that produces clean
     numbers for §5.3–§5.6.

     Existing 5_code/benchmark.py (422 lines) almost works but has the wrong
     shape: 3 scenarios × 10 uniform seeds × 4 methods, no shared compute
     budget, no MSE, no random baseline, hparams not matched to the §0c sweep
     medians, NM run on both losses (asymmetric). Rewrite as a flat ~200-line
     script following the hparam_sweep.py pattern (feedback_keep_it_simple.md).

     Design (locked)

     ┌──────────────┬──────────────────────────────────────────────────────────┐
     │              │                                                          │
     ├──────────────┼──────────────────────────────────────────────────────────┤
     │ Scenarios    │ tonic, adaptation, initial_bursting from                 │
     │ (3)          │ NAUD_PARAMETERS. Same 6 trainable params for all; a, b,  │
     │              │ tau_w fixed at GT.                                       │
     ├──────────────┼──────────────────────────────────────────────────────────┤
     │ Trainable    │ ["C_m", "g_L", "v_reset", "v_T", "E_L", "delta_T"]       │
     │ params       │                                                          │
     ├──────────────┼──────────────────────────────────────────────────────────┤
     │ Inits per    │ 20/bin × 5 bins = 100, stratified by normalised distance │
     │ scenario     │  from GT, fixed seed (per scenario). Same 100 inits hit  │
     │              │ every method within a scenario.                          │
     ├──────────────┼──────────────────────────────────────────────────────────┤
     │ Methods (5)  │ grad_mse, grad_guarino, grad_vanrossum, nm_guarino,      │
     │              │ random                                                   │
     ├──────────────┼──────────────────────────────────────────────────────────┤
     │ Total runs   │ 3 scenarios × 100 inits × 5 methods = 1500 runs          │
     ├──────────────┼──────────────────────────────────────────────────────────┤
     │              │ 500 forward-sim-equivalents per run. Gradient: 250 Adam  │
     │ Budget       │ steps (1 step ≈ 2 sims). NM: maxfev=500. Random: 500     │
     │              │ uniform samples, return best.                            │
     ├──────────────┼──────────────────────────────────────────────────────────┤
     │ Per-method   │ Flat METHOD_CONFIGS dict at top of file (see below).     │
     │ hparams      │                                                          │
     ├──────────────┼──────────────────────────────────────────────────────────┤
     │ Trace        │ dt_ms=0.025, t_max_ms=100, step current I =              │
     │              │ scenario_params["I"] from 5–100 ms.                      │
     └──────────────┴──────────────────────────────────────────────────────────┘

     File: 5_code/benchmark.py (rewrite, target ~200 lines, flat)

     Constants block (top of file)

     Reuse from existing script (lines 36-46): DT_MS, T_MAX_MS, STIM_DELAY_MS,
     STIM_DUR_MS, STIM_END_IDX, STIM_START_IDX, T, N, SPIKE_PEAK_MV. Reuse
     TRAINABLE (line 48).

     Add:
     N_INITS_PER_BIN = 20
     N_BINS = 5
     N_INITS = N_INITS_PER_BIN * N_BINS  # 100
     BUDGET_SIMS = 500
     N_GRAD_STEPS = 250  # 1 grad step ≈ 2 sims
     INIT_SEED_BASE = 0  # per-scenario seed = INIT_SEED_BASE + 
     hash(scenario_name)

     SCENARIOS = {
         "tonic":            {**NAUD_PARAMETERS["tonic"]},
         "adaptation":       {**NAUD_PARAMETERS["adaptation"]},
         "initial_bursting": {**NAUD_PARAMETERS["initial_bursting"]},
     }

     METHODS = ["grad_mse", "grad_guarino", "grad_vanrossum", "nm_guarino",
     "random"]

     METHOD_CONFIGS (flat dict, edit-in-place)

     # Pre-filled with §0c sweep medians where applicable; user tunes in-place 
     before launch.
     METHOD_CONFIGS = {
         "grad_mse": {
             "loss_type": "mse",
             "lr": 0.05,
             "polyak_alpha": 1.0,
             "polyak_beta": 0.75,
             "surrogate_slope": 5.0,
             "loss_kwargs": {"normalize": True},
         },
         "grad_guarino": {
             "loss_type": "guarino",
             "lr": 0.1,
             "polyak_alpha": 1.0,
             "polyak_beta": 0.75,
             "surrogate_slope": 5.0,
             "temperature": 0.3,
             "beta": 7.0,
         },
         "grad_vanrossum": {
             "loss_type": "van_rossum",
             "lr": 0.1,
             "polyak_alpha": 1.0,
             "polyak_beta": 0.75,
             "surrogate_slope": 5.0,
             "loss_kwargs": {"tau_ms": 10.0},
         },
         "nm_guarino": {
             "loss_type": "guarino",
             "temperature": 0.3,
             "beta": 7.0,
             "nm_options": {"adaptive": True, "xatol": 1e-4, "fatol": 1e-4},
         },
         "random": {
             "loss_type": "guarino",      # match NM's loss for apples-to-apples
      chance floor
             "temperature": 0.3,
             "beta": 7.0,
         },
     }

     Shared gradient params (Adam with Polyak normalisation, no LR schedule, no
     early stopping, gradient clip 1.0, no return-best, sigmoid surrogate,
     sigmoid param transform) live in a _grad_training_config(method_cfg) helper
      that builds a TrainingConfig from the method's dict.

     Reused helpers (keep as-is, copy from existing script)

     - _silence() (lines 76-80)
     - _build_current_trace(I_pA) (lines 83-86)
     - simulate_ground_truth(scenario_params) (lines 89-106) — keep, runs once
     per scenario.
     - evaluate(final_params, gt_spike_times, scenario_params) (lines 193-207).

     New helper: stratified init sampling

     def sample_stratified_inits(seed):
         rng = np.random.default_rng(seed)
         gt = NAUD_PARAMETERS["tonic"]  # same param space across scenarios 
     since TRAINABLE is fixed
         inits = []
         for bin_idx in range(N_BINS):
             d_lo, d_hi = bin_idx / N_BINS, (bin_idx + 1) / N_BINS
             accepted = 0
             while accepted < N_INITS_PER_BIN:
                 cand = {n: float(rng.uniform(PARAM_BOUNDS[n].min,
     PARAM_BOUNDS[n].max))
                         for n in TRAINABLE}
                 d = np.mean([abs(cand[n] - gt[n]) / (PARAM_BOUNDS[n].max -
     PARAM_BOUNDS[n].min)
                              for n in TRAINABLE])
                 if d_lo <= d < d_hi:
                     inits.append({"params": cand, "init_dist": float(d), "bin":
      bin_idx})
                     accepted += 1
         return inits
     Note: init_dist is computed against tonic GT for all scenarios (the param
     space is the same since TRAINABLE is fixed). For adaptation/bursting,
     init_dist should be recomputed against that scenario's GT for logging —
     small extension: compute and store init_dist_<scenario> per scenario, or
     sample inits per scenario with gt = SCENARIOS[scenario_name]. Decision: 
     sample inits per scenario so each scenario gets its own well-stratified set
      against its own GT. Cost: 3× the sampling work (negligible) and 3
     different seeds.

     Refactor run_gradient(scenario_params, init_params, method_cfg, gt)

     Take the full method_cfg dict; loss_type selects MSE / Guarino / Van Rossum
      branch. Build TrainingConfig from method_cfg (lr, polyak_alpha,
     polyak_beta, surrogate_slope, n_epochs=N_GRAD_STEPS, return_best=False,
     clip_to_bounds=False, use_param_transform=True, gradient_clip=1.0).

     Pass loss-specific knobs (temperature, beta, loss_kwargs) into
     make_*_loss_fn. MSE branch uses make_mse_loss_fn with
     MSELossConfig(**loss_kwargs).

     Refactor run_nelder_mead(scenario_params, init_params, method_cfg, gt)

     Drop the nm_vanrossum variant — only Guarino. Set options={"maxfev": 
     BUDGET_SIMS, **method_cfg["nm_options"]} (replacing maxiter=2000).

     New: run_random(scenario_params, init_params, method_cfg, gt)

     Sample BUDGET_SIMS uniform vectors within PARAM_BOUNDS using a seed derived
      from the init index (so each "init slot" gets a different random search).
     Evaluate Guarino loss on each via the same JIT'd loss_fn as NM. Return the
     argmin's params + best loss + n_iter=BUDGET_SIMS. The init_params arg is
     ignored at the start but is logged so the row aligns with the others; in
     §4.3 prose: "random search ignores the starting point but is paired with
     each init slot for matched logging".

     Main loop

     def benchmark():
         completed = load_completed_run_ids()  # from existing pattern, lines 
     254-258
         with open(RESULTS_PATH, "a", newline="") as f:
             writer = csv.DictWriter(f, fieldnames=csv_fieldnames())
             if write_header: writer.writeheader()
             for scenario_name, scenario_params in SCENARIOS.items():
                 gt = simulate_ground_truth(scenario_params)
                 n_spikes_gt = len(gt[0])
                 print(f"\n=== {scenario_name} ({n_spikes_gt} GT spikes) ===")
                 seed = INIT_SEED_BASE + abs(hash(scenario_name)) % 10000
                 inits = sample_stratified_inits_for(scenario_params, seed)
                 for init_idx, init in enumerate(inits):
                     for method_name in METHODS:
                         run_id =
     f"{scenario_name}_{init_idx:03d}_{method_name}"
                         if run_id in completed: continue
                         method_cfg = METHOD_CONFIGS[method_name]
                         t0 = time.time()
                         try:
                             r = dispatch(method_name, scenario_params,
     init["params"],
                                          method_cfg, gt)
                             gamma, n_spikes = evaluate(r["final_params"],
     gt[0], scenario_params)
                             writer.writerow(make_row(scenario_name, init_idx,
     init,
                                                      method_name, r, gamma,
     n_spikes,
                                                      n_spikes_gt,
     scenario_params,
                                                      time.time() - t0))
                             f.flush()
                             print(f"{run_id} Γ={gamma:.3f}
     t={time.time()-t0:.1f}s")
                         except Exception as e:
                             print(f"{run_id} ERROR: {e}")

     if __name__ == "__main__":
         benchmark()

     Drop the existing analysis() function (lines 327-417). Move to
     5_code/notebooks/benchmark_analysis.ipynb (Day 2 task).

     CSV schema

     Adapt csv_fieldnames() (lines 212-218):
     - Keep: scenario, method, final_gamma, n_spikes_gt, n_spikes_sim, 
     final_loss, initial_loss, n_iter, wall_time_s, converged, init_params_json,
      final_params_json, gt_params_json, timestamp_iso, norm_err_<param>×6.
     - Add: init_idx (0–99), init_dist (float, against scenario's own GT),
     init_bin (0–4).
     - Drop: seed (subsumed by init_idx), best_epoch (no return-best).
     - run_id schema: {scenario}_{init_idx:03d}_{method}.

     Per feedback_store_dont_reconstruct.md: log everything per row.

     Critical files

     - Modify:
     /Users/paulmayer/Projects/university/40_thesis/5_code/benchmark.py (rewrite
      ~200 lines)
     - Read for APIs (no edits):
       - 5_code/ADoptEX/loss/__init__.py — exports for all three loss factories
       - 5_code/ADoptEX/loss/mse.py:79-137 — MSE signature, MSELossConfig
       - 5_code/ADoptEX/loss/guarino.py:760-830 — Guarino signature
       - 5_code/ADoptEX/loss/van_rossum.py:133-192 — Van Rossum signature,
     VanRossumLossConfig
       - 5_code/ADoptEX/training/trainer.py:438-660 — train() + TrainingConfig
     fields (lr, polyak_alpha, polyak_beta, surrogate_slope, n_epochs,
     gradient_clip, use_param_transform, return_best, clip_to_bounds)
       - 5_code/ADoptEX/core/parameters.py — NAUD_PARAMETERS, PARAM_BOUNDS
       - 5_code/ADoptEX/evaluation/coincidence.py:173-231 — coincidence_factor
     - Output: 5_code/results_benchmark.csv (already in git as untracked — back
     up first if it has content; the resume-on-restart logic will append new
     schema rows but a schema mismatch will break analysis, so back up + delete
     is cleanest before launch)

     Verification

     1. Smoke test (5 min): temporarily set N_INITS_PER_BIN = 1, N_BINS = 2,
     restrict SCENARIOS to {"tonic": ...}. Run end-to-end: 1 scenario × 2 inits
     × 5 methods = 10 runs. Verify CSV has 10 rows, no exceptions, all five
     methods produce finite final_loss, init_dist falls in expected bins.
     2. Init distribution check: load the smoke CSV, group by init_bin, confirm
     init_dist per bin lies in [bin/5, (bin+1)/5).
     3. Budget audit: print n_iter for one row per method. Expect ≈ 250 for
     grad, ≤ 500 for NM, exactly 500 for random.
     4. Hparam reach-through: edit METHOD_CONFIGS["grad_guarino"]["lr"] to 0.99,
      run smoke test, confirm loss explodes — proves the dict actually drives
     training.
     5. Full run: 1500 runs. Estimate 30–60 s/run → 12.5–25 hrs. Launch in
     foreground from 5_code/. CSV flushes per row; resume-on-restart works via
     completed set. Backup: cp results_benchmark.csv 
     results_benchmark_pre_clean.csv && rm results_benchmark.csv before launch.
     6. Sanity on results: random Γ ≤ 0.05 across all bins (chance floor). NM
     should degrade with bin index but recover in close bins. Gradient methods
     sit near random's level → that is the negative result the thesis already
     argues; ship.

     Out of scope (do NOT add)

     - vmap parallelisation across inits (runner.py pattern). Sequential is fine
      and avoids debugging time.
     - LHS init sampling. Stratified rejection sampling matches §0d analysis.
     - soft-DTW. Confirmed dropped.
     - argparse / config classes / external config file. Constants and
     METHOD_CONFIGS at top of file.
     - Plotting in benchmark.py. Notebook owns all plotting.
     - Per-scenario different trainable sets. Confirmed: same 6 params for all
     scenarios.

  Searched for 3 patterns, read 1 file (ctrl+o to expand)

⏺ Now writing benchmark.py.

⏺ Write(benchmark.py)
  ⎿  Added 248 lines, removed 271 lines
       1 +"""Matched-protocol benchmark: 5 methods × 3 scenarios × 100 st
         +ratified inits.                                                
       2 +                                                               
       3 +All methods share a 500 forward-simulation-equivalent budget. H
         +yperparameters                                                 
       4 +are pre-registered per method (see METHOD_CONFIGS). Runs sequen
         +tially, flushes                                                
       5 +CSV per row, resumes on restart via run_id deduplication.      
       6 +"""                                                            
       7 +                                                               
       8  import contextlib
       9  import csv
      10  import json
     ...
      14  from datetime import datetime, timezone
      15  from pathlib import Path
      16  
      10 -import pandas as pd                                            
      11 -import scipy.stats                                             
      12 -                                                               
      17  import jax
      18  import jax.numpy as jnp
      19  import numpy as np
      20  import scipy.optimize
      21  from jax import config
      22  
      19 -from ADoptEX.core import TraceData                             
      20 -from ADoptEX.plotting import trace_stim_window_plot            
      21 -                                                               
      23  config.update("jax_platform_name", "cpu")
      24  
      25  from ADoptEX.core.parameters import NAUD_PARAMETERS, PARAM_BOUN
          DS
      26  from ADoptEX.core.simulation import simulate_jaxley
      27  from ADoptEX.evaluation.coincidence import coincidence_factor
      27 -from ADoptEX.loss import (                                     
      28 -    extract_experimental_features,                             
      29 -    make_guarino_loss_fn,                                      
      30 -    make_van_rossum_loss_fn,                                   
      31 -)                                                              
      28 +from ADoptEX.loss import (MSELossConfig, VanRossumLossConfig,  
      29 +                          extract_experimental_features, make_g
         +uarino_loss_fn,                                                
      30 +                          make_mse_loss_fn, make_van_rossum_los
         +s_fn)                                                          
      31  from ADoptEX.training.trainer import TrainingConfig, setup_trai
          nable_cell, train
      32  
      33  # --- Constants ---
     ...
       40  
       41  STIM_END_IDX = int(round((STIM_DELAY_MS + STIM_DUR_MS) / DT_MS
           ))
       42  STIM_START_IDX = int(round(STIM_DELAY_MS / DT_MS))
       44 -                                                              
       43  T = np.arange(int(round(T_MAX_MS / DT_MS))) * DT_MS
       44  N = len(T)
       45  
       46  TRAINABLE = ["C_m", "g_L", "E_L", "v_T", "delta_T", "v_reset"]
       49 -SEEDS = list(range(10))                                       
       50 -METHODS = ["grad_guarino", "grad_vanrossum", "nm_guarino", "nm
          -_vanrossum"]                                                  
       47  
       48 +N_INITS_PER_BIN = 20                                          
       49 +N_BINS = 5                                                    
       50 +N_INITS = N_INITS_PER_BIN * N_BINS  # 100                     
       51 +BUDGET_SIMS = 500                                             
       52 +N_GRAD_STEPS = 250  # 1 grad step ≈ 2 sims                    
       53 +INIT_SEED_BASE = 0                                            
       54 +                                                              
       55  SCENARIOS = {
       53 -    "tonic": {**NAUD_PARAMETERS["tonic"]},                    
       54 -    "adaptation": {**NAUD_PARAMETERS["adaptation"]},          
       56 +    "tonic":            {**NAUD_PARAMETERS["tonic"]},         
       57 +    "adaptation":       {**NAUD_PARAMETERS["adaptation"]},    
       58      "initial_bursting": {**NAUD_PARAMETERS["initial_bursting"]
           },
       59  }
       60  
       58 -TRAINING_CONFIG = TrainingConfig(                             
       59 -    optimizer="polyak",                                       
       60 -    learning_rate=0.01,                                       
       61 -    n_epochs=200,                                             
       62 -    surrogate_type="sigmoid",                                 
       63 -    surrogate_slope=2,                                        
       64 -    polyak_alpha=1.0,                                         
       65 -    polyak_beta=0.85,                                         
       66 -    clip_to_bounds=False,                                     
       67 -    use_param_transform=True,                                 
       68 -    return_best=True,                                         
       69 -    verbose=False,                                            
       70 -)                                                             
       61 +METHODS = ["grad_mse", "grad_guarino", "grad_vanrossum", "nm_g
          +uarino", "random"]                                            
       62  
       63 +# Per-method hyperparameters (edit in-place to retune; default
          +s from §0c sweep medians).                                    
       64 +METHOD_CONFIGS = {                                            
       65 +    "grad_mse": {                                             
       66 +        "loss_type": "mse",                                   
       67 +        "learning_rate": 0.05,                                
       68 +        "polyak_alpha": 1.0,                                  
       69 +        "polyak_beta": 0.75,                                  
       70 +        "surrogate_slope": 5.0,                               
       71 +        "loss_kwargs": {"normalize": True},                   
       72 +    },                                                        
       73 +    "grad_guarino": {                                         
       74 +        "loss_type": "guarino",                               
       75 +        "learning_rate": 0.1,                                 
       76 +        "polyak_alpha": 1.0,                                  
       77 +        "polyak_beta": 0.75,                                  
       78 +        "surrogate_slope": 5.0,                               
       79 +        "temperature": 0.3,                                   
       80 +        "beta": 7.0,                                          
       81 +    },                                                        
       82 +    "grad_vanrossum": {                                       
       83 +        "loss_type": "van_rossum",                            
       84 +        "learning_rate": 0.1,                                 
       85 +        "polyak_alpha": 1.0,                                  
       86 +        "polyak_beta": 0.75,                                  
       87 +        "surrogate_slope": 5.0,                               
       88 +        "loss_kwargs": {"tau_ms": 10.0},                      
       89 +    },                                                        
       90 +    "nm_guarino": {                                           
       91 +        "loss_type": "guarino",                               
       92 +        "temperature": 0.3,                                   
       93 +        "beta": 7.0,                                          
       94 +        "nm_options": {"adaptive": True, "xatol": 1e-4, "fatol
          +": 1e-4},                                                     
       95 +    },                                                        
       96 +    "random": {                                               
       97 +        "loss_type": "guarino",                               
       98 +        "temperature": 0.3,                                   
       99 +        "beta": 7.0,                                          
      100 +    },                                                        
      101 +}                                                             
      102 +                                                              
      103  RESULTS_PATH = Path(__file__).parent / "results_benchmark.csv"
      104  
      105 +                                                              
      106  # --- Helpers ---
      107  
      108  @contextlib.contextmanager
      109  def _silence():
       78 -    with open(os.devnull, "w") as devnull:                    
       79 -        with contextlib.redirect_stdout(devnull):             
       80 -            yield                                             
      110 +    with open(os.devnull, "w") as devnull, contextlib.redirect
          +_stdout(devnull):                                             
      111 +        yield                                                 
      112  
      113  
      114  def _build_current_trace(I_pA) -> np.ndarray:
       84 -    current = np.zeros(int(round(T_MAX_MS / DT_MS)))          
      115 +    current = np.zeros(N)                                     
      116      current[STIM_START_IDX:STIM_END_IDX] = I_pA
      117      return current
      118  
     ...
      128          idx = int(round(st / DT_MS))
      129          if 0 <= idx < len(sim.voltage):
      130              sim.voltage[idx] = SPIKE_PEAK_MV
      100 -                                                              
      131      exp_features = extract_experimental_features(
      132          voltage_trace=jnp.array(sim.voltage), dt_ms=DT_MS,
      133          stim_duration_ms=STIM_DUR_MS, stim_end_index=STIM_END_
           IDX,
      134      )
      105 -    exp_spike_train = jnp.array(sim.spikes)                   
      106 -    return sim.spike_times, exp_features, exp_spike_train, sim
          -.voltage                                                      
      135 +    return {                                                  
      136 +        "spike_times": sim.spike_times,                       
      137 +        "exp_features": exp_features,                         
      138 +        "exp_spike_train": jnp.array(sim.spikes),             
      139 +        "voltage": jnp.array(sim.voltage),                    
      140 +    }                                                         
      141  
      142  
      109 -def sample_initial_params(seed):                              
      143 +def evaluate(final_params, gt_spike_times, scenario_params):  
      144 +    params = {**scenario_params, **final_params}              
      145 +    with _silence():                                          
      146 +        sim = simulate_jaxley(                                
      147 +            params=params, stim_current_pA=scenario_params["I"
          +],                                                            
      148 +            stim_duration_ms=STIM_DUR_MS, dt_ms=DT_MS,        
      149 +            t_max_ms=T_MAX_MS, stim_delay_ms=STIM_DELAY_MS, us
          +e_surrogate=False,                                            
      150 +        )                                                     
      151 +    if len(sim.spike_times) == 0 or len(gt_spike_times) == 0: 
      152 +        return 0.0, len(sim.spike_times)                      
      153 +    cf = coincidence_factor(                                  
      154 +        spike_times_data=gt_spike_times, spike_times_model=sim
          +.spike_times,                                                 
      155 +        duration_ms=T_MAX_MS, delta_ms=2.0,                   
      156 +    )                                                         
      157 +    return float(cf.gamma), len(sim.spike_times)              
      158 +                                                              
      159 +                                                              
      160 +def sample_stratified_inits(scenario_params, seed):           
      161 +    """Return N_INITS dicts with stratified normalised distanc
          +e from scenario GT."""                                        
      162      rng = np.random.default_rng(seed)
      111 -    return {name: float(rng.uniform(PARAM_BOUNDS[name].min, PA
          -RAM_BOUNDS[name].max))                                        
      112 -            for name in TRAINABLE}                            
      163 +    inits = []                                                
      164 +    for bin_idx in range(N_BINS):                             
      165 +        d_lo, d_hi = bin_idx / N_BINS, (bin_idx + 1) / N_BINS 
      166 +        accepted = 0                                          
      167 +        while accepted < N_INITS_PER_BIN:                     
      168 +            cand = {n: float(rng.uniform(PARAM_BOUNDS[n].min, 
          +PARAM_BOUNDS[n].max))                                         
      169 +                    for n in TRAINABLE}                       
      170 +            d = float(np.mean([                               
      171 +                abs(cand[n] - scenario_params[n]) / (PARAM_BOU
          +NDS[n].max - PARAM_BOUNDS[n].min)                             
      172 +                for n in TRAINABLE                            
      173 +            ]))                                               
      174 +            if d_lo <= d < d_hi:                              
      175 +                inits.append({"params": cand, "init_dist": d, 
          +"bin": bin_idx})                                              
      176 +                accepted += 1                                 
      177 +    return inits                                              
      178  
      179  
      115 -def run_gradient(scenario_params, init_params, loss_type, gt):
      116 -    _, exp_features, exp_spike_train, _ = gt                  
      117 -    cell_params = {**scenario_params, **init_params}          
      118 -    with _silence():                                          
      119 -        cell, data_stimuli, t_max, trainable_handles = setup_t
          -rainable_cell(                                                
      120 -            initial_params=cell_params, current_trace_pA=_buil
          -d_current_trace(scenario_params["I"]),                        
      121 -            dt_ms=DT_MS, config=TRAINING_CONFIG, trainable_par
          -ams=TRAINABLE,                                                
      180 +# --- Loss factory (shared by all methods) ---                
      181 +                                                              
      182 +def _build_loss_fn(method_cfg, cell, data_stimuli, t_max, gt):
      183 +    lt = method_cfg["loss_type"]                              
      184 +    if lt == "mse":                                           
      185 +        return make_mse_loss_fn(                              
      186 +            cell=cell, data_stimuli=data_stimuli, t_max=t_max,
          + dt_ms=DT_MS,                                                 
      187 +            exp_voltage=gt["voltage"], stim_end_index=STIM_END
          +_IDX,                                                         
      188 +            loss_config=MSELossConfig(**method_cfg.get("loss_k
          +wargs", {})),                                                 
      189          )
      123 -    if loss_type == "guarino":                                
      124 -        loss_fn = make_guarino_loss_fn(                       
      190 +    if lt == "guarino":                                       
      191 +        return make_guarino_loss_fn(                          
      192              cell=cell, data_stimuli=data_stimuli, t_max=t_max,
            dt_ms=DT_MS,
      126 -            exp_features=exp_features, stim_duration_ms=STIM_D
          -UR_MS,                                                        
      193 +            exp_features=gt["exp_features"], stim_duration_ms=
          +STIM_DUR_MS,                                                  
      194              stim_end_index=STIM_END_IDX,
      195 +            temperature=method_cfg["temperature"], beta=method
          +_cfg["beta"],                                                 
      196          )
      129 -    else:                                                     
      130 -        loss_fn = make_van_rossum_loss_fn(                    
      197 +    if lt == "van_rossum":                                    
      198 +        return make_van_rossum_loss_fn(                       
      199              cell=cell, data_stimuli=data_stimuli, t_max=t_max,
            dt_ms=DT_MS,
      132 -            exp_spike_train=exp_spike_train, stim_end_index=ST
          -IM_END_IDX,                                                   
      200 +            exp_spike_train=gt["exp_spike_train"], stim_end_in
          +dex=STIM_END_IDX,                                             
      201 +            loss_config=VanRossumLossConfig(**method_cfg.get("
          +loss_kwargs", {})),                                           
      202          )
      134 -    initial_loss = float(loss_fn(trainable_handles))          
      135 -    result = train(loss_fn, trainable_handles, TRAINING_CONFIG
          -)                                                             
      203 +    raise ValueError(f"unknown loss_type: {lt}")              
      204 +                                                              
      205 +                                                              
      206 +def _setup_cell(scenario_params, init_params, surrogate_slope=
          +5.0):                                                         
      207 +    cell_params = {**scenario_params, **init_params}          
      208 +    cfg = TrainingConfig(                                     
      209 +        optimizer="polyak", learning_rate=0.1, n_epochs=N_GRAD
          +_STEPS,                                                       
      210 +        surrogate_type="sigmoid", surrogate_slope=surrogate_sl
          +ope,                                                          
      211 +        clip_to_bounds=False, use_param_transform=True,       
      212 +        return_best=False, verbose=False,                     
      213 +    )                                                         
      214 +    with _silence():                                          
      215 +        return setup_trainable_cell(                          
      216 +            initial_params=cell_params,                       
      217 +            current_trace_pA=_build_current_trace(scenario_par
          +ams["I"]),                                                    
      218 +            dt_ms=DT_MS, config=cfg, trainable_params=TRAINABL
          +E,                                                            
      219 +        )                                                     
      220 +                                                              
      221 +                                                              
      222 +# --- Method runners ---                                      
      223 +                                                              
      224 +def run_gradient(scenario_params, init_params, method_cfg, gt)
          +:                                                             
      225 +    cell, data_stimuli, t_max, handles = _setup_cell(         
      226 +        scenario_params, init_params, method_cfg["surrogate_sl
          +ope"]                                                         
      227 +    )                                                         
      228 +    loss_fn = _build_loss_fn(method_cfg, cell, data_stimuli, t
          +_max, gt)                                                     
      229 +    cfg = TrainingConfig(                                     
      230 +        optimizer="polyak",                                   
      231 +        learning_rate=method_cfg["learning_rate"],            
      232 +        n_epochs=N_GRAD_STEPS,                                
      233 +        surrogate_type="sigmoid",                             
      234 +        surrogate_slope=method_cfg["surrogate_slope"],        
      235 +        polyak_alpha=method_cfg["polyak_alpha"],              
      236 +        polyak_beta=method_cfg["polyak_beta"],                
      237 +        clip_to_bounds=False,                                 
      238 +        use_param_transform=True,                             
      239 +        return_best=False,                                    
      240 +        verbose=False,                                        
      241 +    )                                                         
      242 +    initial_loss = float(loss_fn(handles))                    
      243 +    with _silence():                                          
      244 +        result = train(loss_fn, handles, cfg)                 
      245      return {
      246          "final_params": result.get_params_dict(),
      247          "initial_loss": initial_loss,
      139 -        "final_loss": result.final_loss,                      
      140 -        "best_epoch": result.best_epoch,                      
      248 +        "final_loss": float(result.final_loss),               
      249          "n_iter": len(result.loss_history),
      142 -        "converged": math.isfinite(result.final_loss),        
      250 +        "converged": math.isfinite(float(result.final_loss)), 
      251      }
      252  
      253  
      146 -def run_nelder_mead(scenario_params, init_params, loss_type, g
          -t):                                                           
      147 -    _, exp_features, exp_spike_train, _ = gt                  
      148 -    cell_params = {**scenario_params, **init_params}          
      149 -    with _silence():                                          
      150 -        cell, data_stimuli, t_max, _ = setup_trainable_cell(  
      151 -            initial_params=cell_params, current_trace_pA=_buil
          -d_current_trace(scenario_params["I"]),                        
      152 -            dt_ms=DT_MS, config=TRAINING_CONFIG, trainable_par
          -ams=TRAINABLE,                                                
      153 -        )                                                     
      154 -    if loss_type == "guarino":                                
      155 -        loss_fn = make_guarino_loss_fn(                       
      156 -            cell=cell, data_stimuli=data_stimuli, t_max=t_max,
          - dt_ms=DT_MS,                                                 
      157 -            exp_features=exp_features, stim_duration_ms=STIM_D
          -UR_MS,                                                        
      158 -            stim_end_index=STIM_END_IDX,                      
      159 -        )                                                     
      160 -    else:                                                     
      161 -        loss_fn = make_van_rossum_loss_fn(                    
      162 -            cell=cell, data_stimuli=data_stimuli, t_max=t_max,
          - dt_ms=DT_MS,                                                 
      163 -            exp_spike_train=exp_spike_train, stim_end_index=ST
          -IM_END_IDX,                                                   
      164 -        )                                                     
      165 -    loss_jit = jax.jit(loss_fn)                               
      166 -    jaxley_keys = [list(p.keys())[0] for p in cell.get_paramet
          -ers()]                                                        
      167 -                                                              
      168 -    def nm_loss(x):                                           
      254 +def _scipy_loss_wrapper(loss_jit, jaxley_keys):               
      255 +    def fn(x):                                                
      256          params = [{k: jnp.array([float(x[i])])} for i, k in en
           umerate(jaxley_keys)]
      257          try:
      171 -            val = float(loss_jit(params))                     
      172 -            return val if math.isfinite(val) else 1e6         
      258 +            v = float(loss_jit(params))                       
      259 +            return v if math.isfinite(v) else 1e6             
      260          except Exception:
      261              return 1e6
      262 +    return fn                                                 
      263  
      176 -    x0 = np.array([init_params[name] for name in TRAINABLE])  
      264 +                                                              
      265 +def run_nelder_mead(scenario_params, init_params, method_cfg, 
          +gt):                                                          
      266 +    cell, data_stimuli, t_max, _ = _setup_cell(scenario_params
          +, init_params)                                                
      267 +    loss_fn = _build_loss_fn(method_cfg, cell, data_stimuli, t
          +_max, gt)                                                     
      268 +    loss_jit = jax.jit(loss_fn)                               
      269 +    jaxley_keys = [list(p.keys())[0] for p in cell.get_paramet
          +ers()]                                                        
      270 +    nm_loss = _scipy_loss_wrapper(loss_jit, jaxley_keys)      
      271 +    x0 = np.array([init_params[n] for n in TRAINABLE])        
      272      bounds = [(PARAM_BOUNDS[n].min, PARAM_BOUNDS[n].max) for n
            in TRAINABLE]
      273      initial_loss = nm_loss(x0)
      274      res = scipy.optimize.minimize(
      275          nm_loss, x0, method="Nelder-Mead", bounds=bounds,
      181 -        options={"maxiter": 2000, "adaptive": True, "xatol": 1
          -e-4, "fatol": 1e-4},                                          
      276 +        options={"maxfev": BUDGET_SIMS, **method_cfg["nm_optio
          +ns"]},                                                        
      277      )
      278      return {
      184 -        "final_params": {name: float(res.x[i]) for i, name in 
          -enumerate(TRAINABLE)},                                        
      279 +        "final_params": {n: float(res.x[i]) for i, n in enumer
          +ate(TRAINABLE)},                                              
      280          "initial_loss": initial_loss,
      281          "final_loss": float(res.fun),
      187 -        "best_epoch": -1,                                     
      188 -        "n_iter": res.nfev,                                   
      189 -        "converged": res.success,                             
      282 +        "n_iter": int(res.nfev),                              
      283 +        "converged": bool(res.success),                       
      284      }
      285  
      286  
      193 -def evaluate(final_params, gt_spike_times, scenario_params):  
      194 -    params = {**scenario_params, **final_params}              
      195 -    with _silence():                                          
      196 -        sim = simulate_jaxley(                                
      197 -            params=params, stim_current_pA=scenario_params["I"
          -],                                                            
      198 -            stim_duration_ms=STIM_DUR_MS, dt_ms=DT_MS,        
      199 -            t_max_ms=T_MAX_MS, stim_delay_ms=STIM_DELAY_MS, us
          -e_surrogate=False,                                            
      200 -        )                                                     
      201 -    if len(sim.spike_times) == 0 or len(gt_spike_times) == 0: 
      202 -        return 0.0, len(sim.spike_times)                      
      203 -    cf = coincidence_factor(                                  
      204 -        spike_times_data=gt_spike_times, spike_times_model=sim
          -.spike_times,                                                 
      205 -        duration_ms=T_MAX_MS, delta_ms=2.0,                   
      206 -    )                                                         
      207 -    return float(cf.gamma), len(sim.spike_times)              
      287 +def run_random(scenario_params, init_params, method_cfg, gt, i
          +nit_idx):                                                     
      288 +    cell, data_stimuli, t_max, _ = _setup_cell(scenario_params
          +, init_params)                                                
      289 +    loss_fn = _build_loss_fn(method_cfg, cell, data_stimuli, t
          +_max, gt)                                                     
      290 +    loss_jit = jax.jit(loss_fn)                               
      291 +    jaxley_keys = [list(p.keys())[0] for p in cell.get_paramet
          +ers()]                                                        
      292 +    nm_loss = _scipy_loss_wrapper(loss_jit, jaxley_keys)      
      293 +    rng = np.random.default_rng(10_000 + init_idx)            
      294 +    best_loss, best_x = math.inf, None                        
      295 +    for _ in range(BUDGET_SIMS):                              
      296 +        x = np.array([rng.uniform(PARAM_BOUNDS[n].min, PARAM_B
          +OUNDS[n].max) for n in TRAINABLE])                            
      297 +        v = nm_loss(x)                                        
      298 +        if v < best_loss:                                     
      299 +            best_loss, best_x = v, x                          
      300 +    return {                                                  
      301 +        "final_params": {n: float(best_x[i]) for i, n in enume
          +rate(TRAINABLE)},                                             
      302 +        "initial_loss": float("nan"),                         
      303 +        "final_loss": float(best_loss),                       
      304 +        "n_iter": BUDGET_SIMS,                                
      305 +        "converged": math.isfinite(best_loss),                
      306 +    }                                                         
      307  
      308  
      210 -# --- CSV setup ---                                           
      309 +def dispatch(method_name, scenario_params, init_params, method
          +_cfg, gt, init_idx):                                          
      310 +    if method_name == "random":                               
      311 +        return run_random(scenario_params, init_params, method
          +_cfg, gt, init_idx)                                           
      312 +    if method_name.startswith("nm_"):                         
      313 +        return run_nelder_mead(scenario_params, init_params, m
          +ethod_cfg, gt)                                                
      314 +    return run_gradient(scenario_params, init_params, method_c
          +fg, gt)                                                       
      315  
      316 +                                                              
      317 +# --- CSV ---                                                 
      318 +                                                              
      319  def csv_fieldnames():
      213 -    base = ["run_id", "scenario", "seed", "method", "final_gam
          -ma",                                                          
      214 -            "n_spikes_gt", "n_spikes_sim", "final_loss", "init
          -ial_loss",                                                    
      215 -            "best_epoch", "n_iter", "wall_time_s", "converged"
          -,                                                             
      216 -            "init_params_json", "final_params_json", "gt_param
          -s_json",                                                      
      217 -            "timestamp_iso"]                                  
      320 +    base = ["run_id", "scenario", "init_idx", "init_bin", "ini
          +t_dist", "method",                                            
      321 +            "final_gamma", "n_spikes_gt", "n_spikes_sim",     
      322 +            "final_loss", "initial_loss", "n_iter", "wall_time
          +_s", "converged",                                             
      323 +            "init_params_json", "final_params_json", "gt_param
          +s_json", "timestamp_iso"]                                     
      324      return base + [f"norm_err_{n}" for n in TRAINABLE]
      325  
      326  
      221 -def make_row(scenario_name, seed, method, init_params, r, gamm
          -a, n_spikes_sim, n_spikes_gt,                                 
      222 -             scenario_params, wall_time):                     
      327 +def make_row(scenario_name, init_idx, init, method, r, gamma, 
          +n_spikes_sim,                                                 
      328 +             n_spikes_gt, scenario_params, wall):             
      329      row = {
      224 -        "run_id": f"{scenario_name}_{seed:02d}_{method}",     
      330 +        "run_id": f"{scenario_name}_{init_idx:03d}_{method}", 
      331          "scenario": scenario_name,
      226 -        "seed": seed,                                         
      332 +        "init_idx": init_idx,                                 
      333 +        "init_bin": init["bin"],                              
      334 +        "init_dist": init["init_dist"],                       
      335          "method": method,
      336          "final_gamma": gamma,
      337          "n_spikes_gt": n_spikes_gt,
      338          "n_spikes_sim": n_spikes_sim,
      339          "final_loss": r["final_loss"],
      340          "initial_loss": r["initial_loss"],
      233 -        "best_epoch": r["best_epoch"],                        
      341          "n_iter": r["n_iter"],
      235 -        "wall_time_s": wall_time,                             
      342 +        "wall_time_s": wall,                                  
      343          "converged": r["converged"],
      237 -        "init_params_json": json.dumps(init_params),          
      344 +        "init_params_json": json.dumps(init["params"]),       
      345          "final_params_json": json.dumps(r["final_params"]),
      239 -        "gt_params_json": json.dumps({k: scenario_params[k] fo
          -r k in TRAINABLE if k in scenario_params}),                   
      346 +        "gt_params_json": json.dumps({k: scenario_params[k] fo
          +r k in TRAINABLE}),                                           
      347          "timestamp_iso": datetime.now(timezone.utc).isoformat(
           ),
      348      }
      242 -    for name in TRAINABLE:                                    
      243 -        gt_val = scenario_params.get(name, float("nan"))      
      244 -        final_val = r["final_params"].get(name, float("nan")) 
      245 -        bounds = PARAM_BOUNDS[name]                           
      246 -        row[f"norm_err_{name}"] = abs(final_val - gt_val) / (b
          -ounds.max - bounds.min)                                       
      349 +    for n in TRAINABLE:                                       
      350 +        gt_v = scenario_params[n]                             
      351 +        fv = r["final_params"].get(n, float("nan"))           
      352 +        row[f"norm_err_{n}"] = abs(fv - gt_v) / (PARAM_BOUNDS[
          +n].max - PARAM_BOUNDS[n].min)                                 
      353      return row
      354  
      355  
      356  # --- Main ---
      357  
      358  def benchmark():
      253 -    # Load completed run IDs for resume                       
      359      completed = set()
      360      if RESULTS_PATH.exists():
      361          with open(RESULTS_PATH) as f:
      257 -            for row in csv.DictReader(f):                     
      258 -                completed.add(row["run_id"])                  
      259 -                                                              
      260 -    fieldnames = csv_fieldnames()                             
      362 +            completed = {row["run_id"] for row in csv.DictRead
          +er(f)}                                                        
      363      write_header = not RESULTS_PATH.exists()
      364  
      365      with open(RESULTS_PATH, "a", newline="") as f:
      264 -        writer = csv.DictWriter(f, fieldnames=fieldnames)     
      366 +        writer = csv.DictWriter(f, fieldnames=csv_fieldnames()
          +)                                                             
      367          if write_header:
      368              writer.writeheader()
      267 -                                                              
      369          for scenario_name, scenario_params in SCENARIOS.items(
           ):
      370              gt = simulate_ground_truth(scenario_params)
      270 -            n_spikes_gt = len(gt[0])                          
      271 -            print(f"\n=== {scenario_name} ({n_spikes_gt} GT sp
          -ikes) ===")                                                   
      272 -                                                              
      273 -            gt_params_str = "  GT: " + "  ".join(             
      274 -                f"{k}={scenario_params[k]:.2g}" for k in TRAIN
          -ABLE                                                          
      275 -            )                                                 
      276 -            print(gt_params_str)                              
      277 -                                                              
      278 -            gt_trace = TraceData(                             
      279 -                time=T,                                       
      280 -                voltage=gt[3][:N],                            
      281 -                current=_build_current_trace(scenario_params["
          -I"]),                                                         
      282 -                dt_ms=DT_MS,                                  
      283 -                spike_times=gt[0],                            
      284 -                stim_start_idx=STIM_START_IDX,                
      285 -                stim_end_idx=STIM_END_IDX,                    
      286 -                stim_current_pA=scenario_params["I"],         
      287 -            )                                                 
      288 -            fig = trace_stim_window_plot(gt_trace)            
      289 -            fig.show()                                        
      290 -                                                              
      291 -            for seed in SEEDS:                                
      292 -                init_params = sample_initial_params(seed)     
      293 -                init_str = "  ".join(f"{k}={v:.2g}" for k, v i
          -n init_params.items())                                        
      294 -                print(f"\n  seed={seed}  init: {init_str}")   
      295 -                for method in METHODS:                        
      296 -                    run_id = f"{scenario_name}_{seed:02d}_{met
          -hod}"                                                         
      371 +            n_spikes_gt = len(gt["spike_times"])              
      372 +            seed = INIT_SEED_BASE + abs(hash(scenario_name)) %
          + 10000                                                        
      373 +            inits = sample_stratified_inits(scenario_params, s
          +eed)                                                          
      374 +            print(f"\n=== {scenario_name} ({n_spikes_gt} GT sp
          +ikes, {len(inits)} inits) ===")                               
      375 +            for init_idx, init in enumerate(inits):           
      376 +                for method_name in METHODS:                   
      377 +                    run_id = f"{scenario_name}_{init_idx:03d}_
          +{method_name}"                                                
      378                      if run_id in completed:
      298 -                        print(f"    skip {run_id}")           
      379                          continue
      300 -                                                              
      301 -                    print(f"    {run_id} ... ", end="", flush=
          -True)                                                         
      380 +                    method_cfg = METHOD_CONFIGS[method_name]  
      381                      t0 = time.time()
      382                      try:
      304 -                        if method == "grad_guarino":          
      305 -                            r = run_gradient(scenario_params, 
          -init_params, "guarino", gt)                                   
      306 -                        elif method == "grad_vanrossum":      
      307 -                            r = run_gradient(scenario_params, 
          -init_params, "van_rossum", gt)                                
      308 -                        elif method == "nm_guarino":          
      309 -                            r = run_nelder_mead(scenario_param
          -s, init_params, "guarino", gt)                                
      310 -                        else:                                 
      311 -                            r = run_nelder_mead(scenario_param
          -s, init_params, "van_rossum", gt)                             
      312 -                                                              
      313 -                        gamma, n_spikes_sim = evaluate(r["fina
          -l_params"], gt[0], scenario_params)                           
      383 +                        r = dispatch(method_name, scenario_par
          +ams, init["params"],                                          
      384 +                                     method_cfg, gt, init_idx)
      385 +                        gamma, n_sim = evaluate(r["final_param
          +s"], gt["spike_times"],                                       
      386 +                                                scenario_param
          +s)                                                            
      387                          wall = time.time() - t0
      315 -                        row = make_row(scenario_name, seed, me
          -thod, init_params, r,                                         
      316 -                                       gamma, n_spikes_sim, n_
          -spikes_gt,                                                    
      317 -                                       scenario_params, wall) 
      318 -                        writer.writerow(row)                  
      388 +                        writer.writerow(make_row(scenario_name
          +, init_idx, init,                                             
      389 +                                                 method_name, 
          +r, gamma, n_sim,                                              
      390 +                                                 n_spikes_gt, 
          +scenario_params, wall))                                       
      391                          f.flush()
      320 -                        final_str = "  ".join(f"{k}={r['final_
          -params'].get(k, float('nan')):.2g}"                           
      321 -                                              for k in TRAINAB
          -LE)                                                           
      322 -                        print(f"Γ={gamma:.3f}  {wall:.1f}s  fi
          -nal: {final_str}")                                            
      392 +                        print(f"{run_id} bin={init['bin']} d={
          +init['init_dist']:.2f} "                                      
      393 +                              f"Γ={gamma:.3f} t={wall:.1f}s") 
      394                      except Exception as e:
      324 -                        print(f"ERROR: {e}")                  
      395 +                        print(f"{run_id} ERROR: {e}")         
      396  
      397  
      327 -def analysis():                                               
      328 -    if not RESULTS_PATH.exists():                             
      329 -        print("No results file found. Run benchmark() first.")
      330 -        return                                                
      331 -                                                              
      332 -    df = pd.read_csv(RESULTS_PATH)                            
      333 -    print(f"\n{'='*60}")                                      
      334 -    print(f"ANALYSIS — {len(df)} runs in {RESULTS_PATH.name}")
      335 -    print(f"{'='*60}")                                        
      336 -                                                              
      337 -    # Completeness check                                      
      338 -    counts = df.groupby(["scenario", "method"]).size().unstack
          -(fill_value=0)                                                
      339 -    print(f"\nRun counts per (scenario, method):\n{counts.to_s
          -tring()}")                                                    
      340 -    n_errors = df["final_gamma"].isna().sum()                 
      341 -    if n_errors:                                              
      342 -        print(f"WARNING: {n_errors} runs with NaN gamma")     
      343 -                                                              
      344 -    # Build paired DataFrames for Wilcoxon tests              
      345 -    def paired_gamma(method_a, method_b):                     
      346 -        a = df[df["method"] == method_a].set_index(["scenario"
          -, "seed"])["final_gamma"]                                     
      347 -        b = df[df["method"] == method_b].set_index(["scenario"
          -, "seed"])["final_gamma"]                                     
      348 -        joined = pd.concat([a, b], axis=1, keys=[method_a, met
          -hod_b]).dropna()                                              
      349 -        return joined[method_a].values, joined[method_b].value
          -s, len(joined)                                                
      350 -                                                              
      351 -    comparisons = [                                           
      352 -        ("grad_guarino",  "grad_vanrossum", "Guarino-grad vs V
          -anRossum-grad  (loss function)"),                             
      353 -        ("grad_guarino",  "nm_guarino",     "Guarino-grad vs G
          -uarino-NM       (optimizer)"),                                
      354 -        ("grad_vanrossum","nm_vanrossum",   "VanRossum-grad vs
          - VanRossum-NM   (optimizer)"),                                
      355 -    ]                                                         
      356 -    n_comparisons = len(comparisons)                          
      357 -                                                              
      358 -    print(f"\n{'─'*60}")                                      
      359 -    print("WILCOXON SIGNED-RANK TESTS (paired by scenario+seed
          -)")                                                           
      360 -    print(f"Bonferroni correction: α = 0.05 / {n_comparisons} 
          -= {0.05/n_comparisons:.4f}")                                  
      361 -    print(f"{'─'*60}")                                        
      362 -                                                              
      363 -    stats_rows = []                                           
      364 -    for a_name, b_name, label in comparisons:                 
      365 -        a_vals, b_vals, n = paired_gamma(a_name, b_name)      
      366 -        if n < 2:                                             
      367 -            print(f"\n{label}: not enough paired data (n={n})"
          -)                                                             
      368 -            continue                                          
      369 -                                                              
      370 -        ties = int(np.sum(a_vals == b_vals))                  
      371 -        zero_method = "pratt" if ties / n > 0.2 else "wilcox" 
      372 -        stat, p_raw = scipy.stats.wilcoxon(a_vals, b_vals,    
      373 -                                            zero_method=zero_m
          -ethod,                                                        
      374 -                                            alternative="two-s
          -ided")                                                        
      375 -        p_bonf = min(p_raw * n_comparisons, 1.0)              
      376 -                                                              
      377 -        # Rank-biserial correlation (Kerby 2014)              
      378 -        diffs = a_vals - b_vals                               
      379 -        nonzero = diffs[diffs != 0]                           
      380 -        if len(nonzero) > 0:                                  
      381 -            ranks = scipy.stats.rankdata(np.abs(nonzero))     
      382 -            r_rb = (np.sum(ranks[nonzero > 0]) - np.sum(ranks[
          -nonzero < 0])) / np.sum(ranks)                                
      383 -        else:                                                 
      384 -            r_rb = 0.0                                        
      385 -                                                              
      386 -        med_a = float(np.median(a_vals))                      
      387 -        med_b = float(np.median(b_vals))                      
      388 -                                                              
      389 -        print(f"\n{label}")                                   
      390 -        print(f"  n={n}, ties={ties} (zero_method='{zero_metho
          -d}')")                                                        
      391 -        print(f"  median Γ: {a_name.split('_',1)[1]}={med_a:.3
          -f}  {b_name.split('_',1)[1]}={med_b:.3f}")                    
      392 -        print(f"  W={stat:.1f}  p_raw={p_raw:.4f}  p_bonf={p_b
          -onf:.4f}  r_rb={r_rb:+.3f}")                                  
      393 -        sig = "✓ significant" if p_bonf < 0.05 else "✗ not sig
          -nificant"                                                     
      394 -        print(f"  → {sig} (Bonferroni α=0.05)")               
      395 -                                                              
      396 -        stats_rows.append({                                   
      397 -            "comparison": label, "n": n, "ties": ties, "zero_m
          -ethod": zero_method,                                          
      398 -            "median_a": med_a, "median_b": med_b,             
      399 -            "W": stat, "p_raw": p_raw, "p_bonf": p_bonf, "r_rb
          -": r_rb,                                                      
      400 -        })                                                    
      401 -                                                              
      402 -    # Save stats CSV                                          
      403 -    if stats_rows:                                            
      404 -        stats_path = RESULTS_PATH.with_name("results_benchmark
          -_stats.csv")                                                  
      405 -        pd.DataFrame(stats_rows).to_csv(stats_path, index=Fals
          -e)                                                            
      406 -        print(f"\nStats saved to {stats_path.name}")          
      407 -                                                              
      408 -    # Per-parameter recovery table                            
      409 -    norm_cols = [f"norm_err_{n}" for n in TRAINABLE]          
      410 -    avail = [c for c in norm_cols if c in df.columns]         
      411 -    if avail:                                                 
      412 -        print(f"\n{'─'*60}")                                  
      413 -        print("MEDIAN NORMALISED PARAMETER ERROR (per method, 
          -all scenarios)")                                              
      414 -        print(f"{'─'*60}")                                    
      415 -        recovery = df.groupby("method")[avail].median()       
      416 -        recovery.columns = [c.replace("norm_err_", "") for c i
          -n recovery.columns]                                           
      417 -        print(recovery.round(3).to_string())                  
      418 -                                                              
      419 -                                                              
      398  if __name__ == "__main__":
      399      benchmark()
      422 -    analysis()                                                
  ⎿  Found 2 new diagnostic issues in 1 file (ctrl+o to expand)

⏺ Update(benchmark.py)
  ⎿  Added 2 lines, removed 2 lines
      111          yield
      112  
      113  
      114 -def _build_current_trace(I_pA) -> np.ndarray:                 
      115 -    current = np.zeros(N)                                     
      114 +def _build_current_trace(I_pA):                               
      115 +    current: np.ndarray = np.zeros(N)                         
      116      current[STIM_START_IDX:STIM_END_IDX] = I_pA
      117      return current
      118    ⎿  Found 2 new diagnostic issues in 1 file (ctrl+o to expand)

  Read 1 file (ctrl+o to expand)

⏺ Update(benchmark.py)
  ⎿  Added 1 line, removed 1 line
      214      with _silence():
      215          return setup_trainable_cell(
      216              initial_params=cell_params,
      217 -            current_trace_pA=_build_current_trace(scenario_par
          -ams["I"]),                                                    
      217 +            current_trace_pA=jnp.asarray(_build_current_trace(
          +scenario_params["I"])),                                       
      218              dt_ms=DT_MS, config=cfg, trainable_params=TRAINABL
           E,
      219          )
      220  
  ⎿  Found 1 new diagnostic issue in 1 file (ctrl+o to expand)

⏺ Update(benchmark.py)
  ⎿  Added 2 lines, removed 1 line
      291      jaxley_keys = [list(p.keys())[0] for p in cell.get_paramet
           ers()]
      292      nm_loss = _scipy_loss_wrapper(loss_jit, jaxley_keys)
      293      rng = np.random.default_rng(10_000 + init_idx)
      294 -    best_loss, best_x = math.inf, None                        
      294 +    best_x = np.array([init_params[n] for n in TRAINABLE])    
      295 +    best_loss = nm_loss(best_x)                               
      296      for _ in range(BUDGET_SIMS):
      297          x = np.array([rng.uniform(PARAM_BOUNDS[n].min, PARAM_B
           OUNDS[n].max) for n in TRAINABLE])
      298          v = nm_loss(x)

⏺ Bash(wc -l /Users/paulmayer/Projects/university/40_thesis/5_code/benchmark.py
      && python -c "import ast;
      ast.parse(open('/Users/paulmayer/Projects/university/40_thesi…)
  ⎿       400 /Users/paulmayer/Projects/university/40_thesis/5_code/benchma
     rk.py
     syntax OK

⏺ Bash(cd /Users/paulmayer/Projects/university/40_thesis/5_code &&
      /Users/paulmayer/Projects/university/40_thesis/.venv/bin/python -c "
      import benchmark…)
  ⎿  Running in the background (↓ to manage)

⏺ Bash(sleep 12 && cat /private/tmp/claude-501/-Users-paulmayer-Projects-univers
      ity-40-thesis/c5092246-8014-478f-83b9-fba5063a1e6d/tasks/bxm3yzhjz.output
      | tail -30)
  ⎿  (No output)

  Read 1 file, listed 1 directory (ctrl+o to expand)

⏺ Bash(cd /Users/paulmayer/Projects/university/40_thesis/5_code && timeout 90
      /Users/paulmayer/Projects/university/40_thesis/.venv/bin/python -c "
      import benchmark…)
  ⎿  (eval):1: command not found: timeout

⏺ Bash(/Users/paulmayer/Projects/university/40_thesis/.venv/bin/python -c "
      import benchmark…)
  ⎿  Running in the background (↓ to manage)
  ⎿  (timeout 3m)

⏺ Bash(sleep 30 && wc -c
      /private/tmp/claude-501/-Users-paulmayer-Projects-university-40-thesis/c50
      92246-8014-478f-83b9-fba5063a1e6d/tasks/befmpl51b.output && cat /pri…)
  ⎿  Error: Blocked: sleep 30 followed by: wc -c 
     /private/tmp/claude-501/-Users-paulmayer-Projects-university-40-thesis/c509
     2246-8014-478f-83b9-fba5063a1e6d/tasks/befmpl51b.output cat 
     /private/tmp/claude-501/-Users-paulmayer-Projects-university-40-thesis/c509
     2246-8014-478f-83b9-fba5063a1e6d/tasks/befmpl51b.output. To wait for a 
     condition, use Monitor with an until-loop (e.g. `until <check>; do sleep 2;
      done`). To wait for a command you started, use run_in_background: true. Do
      not chain shorter sleeps to work around this block.

⏺ Monitor(Wait for smoke import to finish)
  ⎿  Monitor started · task bxk7f4nc2 · timeout 120s

  Read 1 file (ctrl+o to expand)

⏺ Waiting for imports to finish — JAX startup is slow.

✻ Cooked for 16m 7s