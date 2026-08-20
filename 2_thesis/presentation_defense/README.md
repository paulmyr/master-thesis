# Master's Thesis Defence

Beamer slides for the thesis *Surrogate Gradients for Gradient-Based Parameter
Estimation in Simplified Neuron Models* (Paul Mayer, KTH).

## Build

Requires LuaLaTeX (the KTH `kthpq` theme uses `fontspec`/OpenType fonts):

```bash
latexmk -lualatex main.tex
```

or a single pass:

```bash
lualatex main.tex
```

## Structure

`main.tex` is the whole deck. It is designed for a **~25 min defence** followed
by an extended Q&A:

- **The Big Picture** (first ~5 min, general audience): why simulate neurons,
  the tuning problem, the spike-discontinuity obstacle, and the
  surrogate-gradient idea.
- **The Model and the Machinery**: the AdEx equations, the two gradient
  problems (spike reset + leak-driven vanishing), surrogate kernels.
- **Contribution 1**: a differentiable AdEx channel for `Jaxley` — verified
  against Brian2 and ~200× faster than its default runtime.
- **The Core Challenge**: loss-function design (MSE, feature-based, Van Rossum)
  and the loss-landscape geometry.
- **The Verdict**: the 2,700-run recovery-radius benchmark, the subthreshold
  exception, and the three failure-mode diagnosis.
- **Conclusions** and future directions.
- **Appendix (A1–A16)**: reference slides for Q&A — parameter tables, surrogate
  math, the coincidence factor, benchmark configuration, BPTT, loss-landscape
  grids, stabilization, the geometry bridge, runtime detail, dataset/compute.

## Assets

- `kthpq-files/` — the KTH Presentation Quarter theme (Isaac Ren, MIT-licensed;
  see `LICENSE.txt`), including logos, decorative line art and fonts.
- `beamerthemekthpq.sty`, `beamercolorthemecustom.sty` — theme entry points.
- `figures/` — plots copied from the thesis `2_thesis/figures/` directory.

`\graphicspath` is set to look in `figures/` first, then the theme's asset
folder, so figures and logos both resolve without further configuration.
