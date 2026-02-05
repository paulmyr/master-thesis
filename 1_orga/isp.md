# Individual plan

Comment to the student: the individual plan needs to be prepared within a few weeks after project start, and has to be accepted by the examiner. The document should be 3-4 pages long.

## PROJECT INFORMATION

- Title: Surrogate Gradients for Gradient-Based Parameter Estimation in Simplified Neuron Models
- Name and e-mail address: Paul Mayer, pmayer@kth.se
- Examiner: Eric Fransén
- Supervisor: Alexander Kozlov 
- Date: 17.01.26
- Keywords: Automatic Gradient Descent, Adaptive Exponential Integrate-and-Fire Model, Simplified Neuron Model, Surrogate Gradients, Parameter Estimation

## BACKGROUND & OBJECTIVE

This project is conducted within computational neuroscience, specifically focusing on parameter estimation for simplified biophysical neuron models.
Large-scale brain simulations require efficient methods for fitting neuron model parameters to experimental data.
While sophisticated Hodgkin-Huxley-type models can precisely capture membrane behavior, they require enormous computational resources.
On the other hand, simplified neuron models (like the Adaptive Exponential Integrate-and-Fire Model, AdEx) offer drastically lower computational cost while reproducing diverse firing patterns faithfully.
This makes them prime candidates for whole-brain simulations.

The project connects to recent developments in gradient-based optimization for neuron models.
The Jaxley framework demonstrated success using automatic differentiation for HH-type models.
However, it lacks support for simplified models due to their non-differentiable spike mechanism (the discrete reset when membrane potential crosses threshold).
Surrogate gradients, originally developed for training spiking neural networks, offer a potential solution by replacing discontinuous derivatives with smooth approximations during backpropagation.
This requires loss functions to be differentiable as well.
Wheras non-gradient based parameter optimization strategies can rely on non-differentiable feature based loss functions, gradient-based differentiation requires differentiable loss functions.
This opens up for the question on what loss function design is required to achieve successful parameter fittings.

To study loss function design, signal processing techniques might provide valuable insight.
But mainly, this work bridges techniques from machine learning (surrogate gradients) with biophysical modeling, addressing a gap in current tools.
It is of interest to computational neuroscientists working toward large-scale brain simulations where efficient and accurate parameter fitting is essential.

## RESEARCH QUESTION & METHOD

The main question I want to answer is:
"Can gradient-based parameter optimization produce useful parameter updates for the AdEx model?"

Related questions that will also be studied in this project are:
1. What loss function design is required to achieve successful gradient-based optimization for simplified neuron models?
2. How do different surrogate gradient functions affect optimization convergence and parameter quality?
3. How does gradient-based optimization compare to derivative-free methods in terms of accuracy and computational efficiency?

### Objectives: Break down the research questions to measurable objectives.

Thhe following milestones need to be completed in order to answer the research question:
- [x] Implement AdEx model using a bio-physical neuron simulator that allows automatic differentiation.
- [x] Extend AdEx implementation to allow for surrogate gradients.
- [ ] Design differentiable loss function that allows successful gradient based optimization.
- [ ] Implement testing framework that allows optimization on multiple experimental traces.
- [ ] Measure computational efficiency of optimization method
- [ ] Compare gradient-based strategie against feature based state-of-the-art genetic / non-differentiable alternatives.

### Tasks: Describe the tasks that are necessary to reach the objectives. For each task, describe the challenges it involves.
Some of the task have already been started within the HPC project course. For example, I will use my previously implemented AdEx model.

- Building a differentiable loss function will require careful study of existing approaches e.g. in signal processing or machine learning domains.
Implementation will use the JAX framework. A main challenge will be the influence of many hyperparameters that are inherint to optimization problems. Identifing what influence is due to the non-convex paramter space, to the actual optimizers and to the loss function may be difficult.
- Testing framework will be mostly an implementation overhead. I don't expect much difficulty, however, the main challenge may be the result of different data formats or signal lengths --- handling time series data might be challenging.
- Measuring computational efficiency can be highly system and data dependent. Finding good testing metrics might be challenging.
- Finding other state-of-the-art optimizers will not be as easy since there is no standard benchmark set for parameter tuning in neuroscience. Not all code may be easily accessible as well.

### Method: Describe the method/s that will be followed. Explain why they are appropriate for the project or for the specific tasks.

### Ethics and Sustainability: Does the project address questions of ethics or sustainability? Does the project raise ethical or sustainability questions? If yes, how could these be handled?

### Limitations: Define the limitations on what is to be done (so that it is clear what is not included in the degree project).

### Risks: Explain what can go wrong and delay or make the project impossible to conclude. Explain how you will deal with these problems.


## EVALUATION & NEWS VALUE

### Evaluation: How is it determined if the objectives of the degree project have been fulfilled and if the research question has been adequately answered? What kind of qualitative or quantitative measures can be defined and evaluated?

### Expected scientific results: How is the work scientifically relevant? The work's innovation/news value. Why does someone want to read the finished work? And who are these people?

## PRE-STUDY

Description of the literature studies. What areas will the literature study focus on? How shall the necessary knowledge on background and state-of-the-art be obtained? What preliminarily important references have been identified?

## CONDITIONS & SCHEDULE

List of the resources are needed to solve the problem. This can be technical equipment, software, or data, but also experiment and interview subjects.

Describe the way the external supervisor will be involved in the project.

Provide a project timeline, specifying the main tasks and the time allocated for them, milestones (time of achievement of intermediate goals)

## USE OF GENERATIVE AI 

Describe the way you used generative AI for writing your individual plan.

## REFERENCES

List references, including information Authors, Title, Journal/Book/Conference/Website, Publication date


---
# Vorschlag claude:

  ---
  Method: Describe the method/s that will be followed. Explain why
   they are appropriate for the project or for the specific tasks.

  The methodology follows an experimental computational approach
  combining implementation, validation, and systematic evaluation:

  1. Implementation within Jaxley Framework
  The AdEx model will be implemented as a Jaxley channel using the
   JAX framework for automatic differentiation. Surrogate
  gradients will be implemented using JAX's custom_vjp
  functionality, which allows defining custom vector-Jacobian
  products. Three surrogate functions will be explored: sigmoid,
  exponential, and SuperSpike surrogates. This approach is
  appropriate because Jaxley has demonstrated success for HH-type
  models, and extending it to simplified models maintains
  compatibility with existing neuroscience workflows.

  2. Implementation Verification
  The AdEx implementation will be validated against Brian2, a
  widely-used reference simulator. Verification metrics include
  visual comparison of voltage traces, spike count matching, and
  spike timing differences (target: <1ms). This cross-validation
  approach ensures implementation correctness before proceeding to
   optimization experiments.

  3. Loss Function Development
  Two loss function approaches will be investigated:
  - MSE-based loss on raw voltage traces (baseline)
  - Feature-based loss adapted from Guarino et al. [9], capturing
  spike timing, interspike intervals, firing rate, and
  subthreshold behavior

  The feature-based approach requires developing differentiable
  approximations of discrete spike detection operations using soft
   thresholds and weighted averages. This is necessary because
  gradient-based optimization requires end-to-end
  differentiability.

  4. Systematic Evaluation
  Parameter optimization will be evaluated using intracellular
  recordings from striatal projection neurons. Performance will be
   assessed through:
  - Visual comparison of fitted vs. experimental traces
  - Coincidence factor Γ for spike timing precision (benchmark: Γ
  > 0.5)
  - Computational cost measurements

  5. Comparative Analysis
  Gradient-based results will be compared against derivative-free
  baselines (genetic algorithms, grid search) to assess relative
  performance and computational efficiency.

  This methodology is appropriate because it follows a systematic
  progression from implementation to validation to evaluation,
  allowing identification of specific failure modes and enabling
  iterative improvement of the loss function design.

  Ethics and Sustainability: Does the project address questions of
   ethics or sustainability? Does the project raise ethical or
  sustainability questions? If yes, how could these be handled?

  Ethics:
  This project uses publicly available experimental data from
  prior research (Johansson and Silberberg, 2020). No new animal
  experiments are conducted. The computational methods developed
  could ultimately reduce the need for animal experiments by
  enabling better in-silico modeling and prediction of neural
  behavior.

  Sustainability:
  The project addresses computational sustainability in
  neuroscience. Simplified models like AdEx require orders of
  magnitude less computational resources than detailed HH-type
  models. Enabling efficient parameter fitting for these models
  could significantly reduce the energy consumption required for
  large-scale brain simulations. Additionally, gradient-based
  optimization, if successful, would be more computationally
  efficient than evolutionary algorithms that require many more
  simulation evaluations.

  No significant ethical concerns are raised by this work.

  Limitations: Define the limitations on what is to be done (so
  that it is clear what is not included in the degree project).

  The following are explicitly out of scope for this project:

  1. Multi-compartment models: Only single-compartment AdEx
  neurons will be considered. Extension to complex morphologies
  (dendrites, axons) is not included.
  2. Network-level optimization: The project focuses on
  single-neuron parameter fitting. Fitting parameters for networks
   of neurons is not addressed.
  3. All neuron types: Evaluation will focus on striatal
  projection neurons. Comprehensive validation across diverse
  neuron types (cortical, cerebellar, etc.) is limited by
  available data and time.
  4. Hyperparameter optimization: While some hyperparameter
  exploration will be conducted, exhaustive optimization of
  surrogate gradient parameters, learning rates, and loss function
   weights is not the primary focus.
  5. Real-time or embedded applications: The implementation
  targets research use in Python/JAX, not deployment in real-time
  systems.
  6. Comparison with all existing methods: Only selected baseline
  methods (genetic algorithms, grid search) will be compared.
  Comprehensive benchmarking against all published optimization
  approaches is not feasible.
  7. Theoretical analysis: The focus is empirical. Formal
  convergence proofs or theoretical analysis of surrogate gradient
   approximation quality are not included.

  Risks: Explain what can go wrong and delay or make the project
  impossible to conclude. Explain how you will deal with these
  problems.
  Risk: Gradient-based optimization fails to produce useful
    parameters
  Impact: High
  Likelihood: Medium
  Mitigation: The preliminary work already shows this is
    challenging. If optimization consistently fails, the
    project will pivot to documenting failure modes and
    characterizing the loss landscape, which is itself a
    valuable contribution.
  ────────────────────────────────────────
  Risk: Non-convex loss landscape prevents convergence
  Impact: Medium
  Likelihood: High
  Mitigation: Implement multiple initialization strategies,
    learning rate schedules, and gradient clipping.
    Explore different surrogate gradient functions and
    hyperparameters. Document which conditions enable vs.
     prevent convergence.
  ────────────────────────────────────────
  Risk: Differentiable feature extraction introduces bias
  Impact: Medium
  Likelihood: Medium
  Mitigation: Validate soft feature approximations against hard
    (non-differentiable) counterparts. Quantify
    approximation errors and their impact on
    optimization.
  ────────────────────────────────────────
  Risk: Limited experimental data availability
  Impact: Medium
  Likelihood: Low
  Mitigation: The Johansson and Silberberg dataset is already
    available. If additional data is needed, publicly
    available datasets from the Allen Brain Atlas or
    other sources can be used.
  ────────────────────────────────────────
  Risk: Computational resource constraints
  Impact: Low
  Likelihood: Low
  Mitigation: The project uses simplified models specifically
    because they are computationally efficient. Standard
    laptop hardware is sufficient for all experiments.
  ────────────────────────────────────────
  Risk: Software compatibility issues
  Impact: Low
  Likelihood: Medium
  Mitigation: Pin software versions (Jaxley, JAX, Brian2). The
    preliminary implementation already runs successfully.
  Contingency plan: If gradient-based optimization proves
  fundamentally unsuitable for AdEx parameter estimation, the
  project can still contribute by (1) documenting why this
  approach fails, (2) characterizing loss function requirements,
  and (3) providing a validated, differentiable AdEx
  implementation for future research.

  EVALUATION & NEWS VALUE

  Evaluation: How is it determined if the objectives of the degree
   project have been fulfilled and if the research question has
  been adequately answered? What kind of qualitative or
  quantitative measures can be defined and evaluated?

  The project objectives will be evaluated using the following
  criteria:

  Quantitative Measures:

  1. Coincidence Factor (Γ): The primary metric for parameter
  fitting quality. Success threshold: Γ > 0.5, matching or
  approaching the Γ ≈ 0.82-0.83 achieved by Jolivet et al. for
  well-fitted AdEx models.
  2. Spike count accuracy: Percentage of fitted models producing
  the correct number of spikes (±1 spike tolerance).
  3. First spike timing error: Mean absolute error in predicting
  the first spike time (target: <5ms).
  4. Convergence rate: Number of optimization iterations required
  to reach a stable loss value.
  5. Computational efficiency: Wall-clock time per optimization
  run compared to genetic algorithm baselines.

  Qualitative Measures:

  1. Loss function characterization: Documentation of which loss
  function designs enable vs. prevent successful optimization.
  2. Surrogate gradient comparison: Qualitative assessment of how
  different surrogate functions affect optimization behavior.
  3. Failure mode analysis: Identification and documentation of
  systematic failure patterns (e.g., convergence to non-spiking
  solutions).

  Success Criteria:

  - Full success: Gradient-based optimization achieves Γ > 0.5
  with computational advantage over baselines.
  - Partial success: Gradient-based optimization produces spiking
  behavior with correct spike count but limited timing precision.
  - Documented failure: Clear characterization of why
  gradient-based optimization fails, with recommendations for
  future work.

  Any of these outcomes constitutes a valid scientific
  contribution.

  Expected scientific results: How is the work scientifically
  relevant? The work's innovation/news value. Why does someone
  want to read the finished work? And who are these people?

  Scientific Relevance:

  This work addresses a methodological gap in computational
  neuroscience: the lack of gradient-based parameter fitting tools
   for simplified neuron models. While Jaxley has demonstrated
  success for HH-type models, simplified models like AdEx remain
  limited to derivative-free optimization methods despite their
  importance for large-scale brain simulations.

  Innovation/News Value:

  1. First systematic study of loss function requirements for
  gradient-based optimization of simplified neuron models in
  biophysical simulation contexts.
  2. Differentiable AdEx implementation in Jaxley, enabling the
  broader community to use gradient-based methods for simplified
  models.
  3. Bridge between SNN training and biophysical fitting:
  Demonstrates whether techniques from machine learning (surrogate
   gradients) can be transferred to neuroscience applications.
  4. Practical guidance on loss function design for researchers
  attempting similar approaches.

  Target Audience:

  1. Computational neuroscientists working on large-scale brain
  simulations who need efficient parameter fitting methods.
  2. Developers of neural simulation frameworks (Jaxley, Brian2,
  NEST) interested in extending automatic differentiation support.
  3. Machine learning researchers working on spiking neural
  networks who may benefit from insights on loss function design.
  4. Researchers in the Grillner/Bhalla labs and others working
  toward whole-brain simulation using simplified models.

  PRE-STUDY

  Literature Study Focus Areas:

  1. Simplified neuron models: Mathematical formulation, parameter
   interpretation, and fitting approaches for AdEx and related
  models (Brette & Gerstner, 2005; Naud et al., 2008).
  2. Surrogate gradient methods: Theory and practice of surrogate
  gradients for spiking neural networks (Neftci et al., 2019;
  Zenke & Ganguli, 2018; Gygax & Zenke, 2025).
  3. Automatic differentiation for neuroscience: Jaxley framework
  and related approaches (Deistler et al., 2025; Jones & Kording,
  2024).
  4. Parameter optimization for neuron models: Evolutionary
  algorithms, grid search, and other derivative-free methods (Van
  Geit et al., 2008; Guarino et al., 2025).
  5. Loss function design: Feature-based losses, spike train
  metrics, and differentiable approximations (Jolivet et al.,
  2008; Guarino et al., 2025).

  Knowledge Acquisition:

  - Review of primary literature in computational neuroscience and
   machine learning
  - Study of Jaxley and JAX documentation and source code
  - Analysis of existing AdEx parameter fitting implementations

  Preliminarily Important References:

  - Deistler et al. (2025) - Jaxley framework
  - Neftci et al. (2019) - Surrogate gradient learning review
  - Brette & Gerstner (2005) - Original AdEx paper
  - Guarino et al. (2025) - Feature-based loss function
  - Jolivet et al. (2008) - Coincidence factor metric
  - Zenke & Ganguli (2018) - SuperSpike surrogate

  CONDITIONS & SCHEDULE

  Resources Required:

  - Hardware: Personal laptop (Apple M1 Silicon) - sufficient for
  all experiments
  - Software: Python 3.14+, JAX, Jaxley 0.12+, Brian2 (for
  validation), standard scientific Python stack
  - Data: Intracellular recordings from Johansson & Silberberg
  (2020) - publicly available
  - Code: Existing implementation from HPC project course (AdEx
  model, surrogate gradients)

  Supervisor Involvement:

  Alexander Kozlov will provide guidance on:
  - Neuroscience domain knowledge and biological interpretation
  - Connection to ongoing striatum modeling work
  - Feedback on experimental design and results interpretation

  Regular meetings (approximately bi-weekly) will be scheduled to
  discuss progress and address challenges.

  Project Timeline:
  ┌──────────────┬────────────────────────────┬──────────────────┐
  │    Period    │           Tasks            │    Milestone     │
  ├──────────────┼────────────────────────────┼──────────────────┤
  │ Weeks 1-2    │ Literature review, refine  │ Pre-study        │
  │ (Jan)        │ research questions         │ complete         │
  ├──────────────┼────────────────────────────┼──────────────────┤
  │ Weeks 3-4    │ Improve differentiable     │                  │
  │ (Feb)        │ loss function              │ Loss function v1 │
  │              │ implementation             │                  │
  ├──────────────┼────────────────────────────┼──────────────────┤
  │ Weeks 5-6    │ Systematic optimization    │                  │
  │ (Feb-Mar)    │ experiments on single      │ Initial results  │
  │              │ trace                      │                  │
  ├──────────────┼────────────────────────────┼──────────────────┤
  │ Weeks 7-8    │ Extend to multiple         │ Testing          │
  │ (Mar)        │ experimental traces        │ framework        │
  │              │                            │ complete         │
  ├──────────────┼────────────────────────────┼──────────────────┤
  │ Weeks 9-10   │ Surrogate gradient         │ Surrogate        │
  │ (Mar-Apr)    │ comparison experiments     │ evaluation       │
  │              │                            │ complete         │
  ├──────────────┼────────────────────────────┼──────────────────┤
  │ Weeks 11-12  │ Baseline comparisons       │ Comparative      │
  │ (Apr)        │ (genetic algorithms)       │ analysis         │
  │              │                            │ complete         │
  ├──────────────┼────────────────────────────┼──────────────────┤
  │ Weeks 13-14  │ Computational efficiency   │ All experiments  │
  │ (Apr-May)    │ measurements               │ complete         │
  ├──────────────┼────────────────────────────┼──────────────────┤
  │ Weeks 15-17  │ Thesis writing             │ Draft complete   │
  │ (May)        │                            │                  │
  ├──────────────┼────────────────────────────┼──────────────────┤
  │ Weeks 18-20  │ Revision and final         │ Thesis submitted │
  │ (Jun)        │ submission                 │                  │
  └──────────────┴────────────────────────────┴──────────────────┘
  USE OF GENERATIVE AI

  Generative AI (Claude, Anthropic) was used in the preparation of
   this individual plan for:

  1. Document structuring: Assistance in organizing and formatting
   the plan according to the template requirements.
  2. Writing refinement: Improving clarity and conciseness of
  technical descriptions.
  3. Consistency checking: Ensuring alignment between the
  individual plan and the project proposal.

  All content was reviewed and validated by the author. The
  scientific content, research questions, and methodology are the
  author's own work, building on preliminary results from the HPC
  project course.

  REFERENCES

  [1] W. Gerstner, Ed., Neuronal Dynamics: From Single Neurons to
  Networks and Models of Cognition. Cambridge University Press,
  2014.

  [2] R. Naud, N. Marcille, C. Clopath, and W. Gerstner, "Firing
  patterns in the adaptive exponential integrate-and-fire model,"
  Biological Cybernetics, vol. 99, no. 4-5, Nov. 2008.

  [3] R. Brette and W. Gerstner, "Adaptive Exponential
  Integrate-and-Fire Model as an Effective Description of Neuronal
   Activity," Journal of Neurophysiology, vol. 94, no. 5, Nov.
  2005.

  [4] M. Deistler, K. L. Kadhim, M. Pals, et al., "Jaxley:
  Differentiable simulation enables large-scale training of
  detailed biophysical models of neural dynamics," Nature Methods,
   Nov. 2025.

  [5] E. O. Neftci, H. Mostafa, and F. Zenke, "Surrogate Gradient
  Learning in Spiking Neural Networks," IEEE Signal Processing
  Magazine, May 2019.

  [6] F. Zenke and S. Ganguli, "SuperSpike: Supervised Learning in
   Multilayer Spiking Neural Networks," Neural Computation, vol.
  30, no. 6, Jun. 2018.

  [7] D. Guarino, I. Carannante, and A. Destexhe, "A unified model
   library maps how neuromodulation reshapes the excitability
  landscape of neurons across the brain," PLOS Computational
  Biology, vol. 21, no. 12, Dec. 2025.

  [8] R. Jolivet, R. Kobayashi, A. Rauch, et al., "A benchmark
  test for a quantitative assessment of simple neuron models,"
  Journal of Neuroscience Methods, vol. 169, no. 2, Apr. 2008.

  [9] W. Van Geit, E. De Schutter, and P. Achard, "Automated
  neuron model optimization techniques: A review," Biological
  Cybernetics, vol. 99, no. 4-5, Nov. 2008.

  [10] J. Gygax and F. Zenke, "Elucidating the Theoretical
  Underpinnings of Surrogate Gradient Learning in Spiking Neural
  Networks," Neural Computation, vol. 37, no. 5, Apr. 2025.

  [11] Y. Johansson and G. Silberberg, "The Functional
  Organization of Cortical and Thalamic Inputs onto Five Types of
  Striatal Neurons Is Determined by Source and Target Cell
  Identities," Cell Reports, vol. 30, no. 4, Jan. 2020.

  [12] I. S. Jones and K. P. Kording, "Efficient optimization of
  ODE neuron models using gradient descent," 2024.

  ---
  This completes the individual plan. Note that the timeline
  assumes a 20-week project period starting in January 2026 -
  adjust dates according to your actual start date.
