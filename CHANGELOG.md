# Changelog

All notable changes to this project are documented in this file. The format
follows Keep a Changelog, and the project adheres to semantic versioning.

## [0.1.0] - 2026-09-17

### Added

- Forward solvers for model (1): a Fourier pseudo-spectral scheme in one and
  two space dimensions with implicit diffusion, explicit nonlocal advection,
  two-thirds dealiasing, an exact exponential memory update and a positivity
  monitor, and an independent positivity-preserving finite-volume scheme for
  the periodic, the no-flux and the masked bounded cases.
- Steady-state solver based on the fixed-point characterisation implied by
  equation (7), with damped iteration, continuation in the aggregation ratio
  and the residual of the stationary identity.
- Linear theory of Section 3: the mode matrix of equation (6), the instability
  criterion of Proposition 3, the index of Corollary 1 and the reconstruction
  of the parameters from two Fourier modes given by Theorem 2.
- Synthetic design of Section 4.2: the parameter values of Table 3, the full
  factorial design of Table 4 with its stable condition indices and identifiers
  and its deterministic per-condition seeds, and the observation model (9).
- Inference machinery: log-uniform priors, Gaussian likelihoods for transient
  and stationary data, profile likelihood with interval classification, an
  adaptive random-walk Metropolis sampler with atomic checkpointing, resume and
  a wall-time guard, an ensemble sampler wrapper with an HDF5 backend, and
  convergence diagnostics based on rank-normalised split R-hat and the bulk
  effective sample size.
- Application to the wolf space-use data of Section 6: schema validation of the
  Movebank data package, cleaning and projection, study-area construction,
  a masked fixed-point solver, the weighted point likelihood of equation (10)
  and model comparison by leave one out and leave one block out
  cross-validation.
- Experiment scripts for Experiments 0.3 to 4.4, Slurm job files and a chaining
  helper for the CSC Roihu supercomputer, figure scripts for Fig. 1 to Fig. 7,
  table scripts for Table 3 to Table 8 and the electronic supplement.
- Pre-experiment checks of the structural results, the parameter design and the
  forward solve cost.
- Test suite covering the kernels, the structural results, the solvers, the
  design, the likelihoods and priors, the checkpointing and the wolf pipeline.
- Policy checkers for the character, comment and licence header rules, together
  with the continuous integration workflow that enforces them.
- Documentation of the model equations, the experiment map, the reproduction
  route, the cluster workflow, the figure style and the coding standards.

### Fixed

- The comment policy checker misreported dictionary keys inside a bracketed
  expression as Python docstrings, and did not exempt shell line continuations
  or the keywords that only close a block.
- The bounded convolution of the finite-volume solver returned an array of the
  wrong length, because the discrete convolution was taken in the "same" mode
  against a kernel longer than the field.
- The masked steady-state solver of the wolf application used the aggregation
  ratio with the dimensions of an area, so that the fitted density was
  numerically indistinguishable from a uniform one. The ratio is now used in
  the dimensionless form kappa u_bar of Proposition 3.
- The bounded convolution of the finite-volume solver evaluated a direct
  discrete convolution at every time step, which dominated the cost of a
  bounded solve. It is now evaluated by a padded fast Fourier transform whose
  kernel transform is computed once per grid and kernel.
- The verification of Proposition 4 on a bounded interval used a strongly
  supercritical aggregation ratio, whose narrow aggregate the first-order
  scheme could not resolve, so that the logarithm in the stationary identity
  was dominated by the discretisation error. The test now uses the same
  twenty per cent supercriticality as the design of Section 4.2, with the onset
  value formed from the cosine modes admitted by the no-flux boundary.
- The two-dimensional inference script and the wolf fitting script imported
  shared helpers from other experiment scripts, which are not an importable
  package, so both failed at import time. The two-dimensional design, the
  study area loader and the summary reader now live in the nmi package.
- The wolf scripts raised an unclear key error when they were given a
  configuration without a wolf block. They now report the mismatch and stop.
- The verification of Proposition 4 compared the stationary identity with a
  constant at a single resolution, which the first-order upwind flux cannot
  satisfy for the top-hat kernel, whose perceived gradient is far steeper than
  that of the Gaussian kernel. The test now checks what the proposition
  asserts at the discrete level, namely that the residual falls when the grid
  is refined from one hundred to two hundred cells.
