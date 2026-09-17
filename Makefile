# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Driver for installation, policy checks, tests, the smoke
# configuration and the generation of every figure and table of the manuscript.
# Manuscript: supports docs/reproducibility.md.
# Inputs: the tracked source tree. Outputs: files under results/.

# Interpreter used by every target, overridable on the command line.
PYTHON ?= python
# Directory that holds the tracked summary outputs read by figures and tables.
SUMMARY_DIR ?= results/summary
# Configuration used by the quick end to end demonstration.
SMOKE_CONFIG ?= configs/smoke.yaml

# Targets that do not correspond to files of the same name.
.PHONY: install lint policy test test-all precheck smoke figures tables online-resource clean

# Install the package and the development extras into the active environment.
install:
	$(PYTHON) -m pip install -e .[dev]  # Editable installation with test and lint tools.

# Run the linter over the whole repository.
lint:
	$(PYTHON) -m ruff check .  # Static checks of style and of unused names.

# Run the three repository policy checks.
policy:
	$(PYTHON) tools/check_text_policy.py  # Forbidden characters in tracked files.
	$(PYTHON) tools/check_comment_policy.py  # Comment coverage of every code line.
	$(PYTHON) tools/check_license_headers.py  # Presence of the licence notice.

# Run the fast part of the test suite.
test:
	$(PYTHON) -m pytest -m "not slow"  # Exclude tests that run full solvers or samplers.

# Run the whole test suite including the slow tests.
test-all:
	$(PYTHON) -m pytest  # Every test, including the slow verification tests.

# Run the pre-experiment checks of the structural results and of the design.
precheck:
	$(PYTHON) precheck/check_proofs.py  # Propositions 1 and 3, Lemma 1, Theorem 2, Corollary 1.
	$(PYTHON) precheck/design_params.py  # Table 3 values and the critical aggregation ratios.
	$(PYTHON) precheck/timing.py  # Forward solve cost at the candidate resolutions.

# Run every stage of the pipeline once on the small smoke configuration.
smoke:
	$(PYTHON) experiments/make_synthetic_data.py --config $(SMOKE_CONFIG)  # Reference data.
	$(PYTHON) experiments/exp_0_3_checkpoint_test.py --config $(SMOKE_CONFIG)  # Checkpoint test.
	$(PYTHON) experiments/exp_1_1_convergence_1d.py --config $(SMOKE_CONFIG)  # Convergence.
	$(PYTHON) experiments/exp_1_2_steady_state.py --config $(SMOKE_CONFIG)  # Steady state.
	$(PYTHON) experiments/exp_1_3_convergence_2d.py --config $(SMOKE_CONFIG)  # Two dimensions.
	$(PYTHON) experiments/exp_2_1_structural_checks.py --config $(SMOKE_CONFIG)  # Structural checks.
	$(PYTHON) experiments/exp_2_2_mcmc_pilot.py --config $(SMOKE_CONFIG)  # Sampler pilot.
	$(PYTHON) experiments/exp_2_3_profile_likelihood.py --config $(SMOKE_CONFIG)  # Profiles.
	$(PYTHON) experiments/exp_2_4_mcmc_1d.py --config $(SMOKE_CONFIG)  # One-dimensional chains.
	$(PYTHON) experiments/exp_2_5_diagnostics.py --config $(SMOKE_CONFIG)  # Convergence diagnostics.
	$(PYTHON) experiments/exp_2_6_kernel_discrimination.py --config $(SMOKE_CONFIG)  # Kernel choice.
	$(PYTHON) experiments/exp_3_1_pilot_2d.py --config $(SMOKE_CONFIG)  # Two-dimensional pilot.
	$(PYTHON) experiments/exp_3_2_mcmc_2d.py --config $(SMOKE_CONFIG)  # Two-dimensional chains.
	$(PYTHON) experiments/exp_3_3_analysis_2d.py --config $(SMOKE_CONFIG)  # Two-dimensional summary.
	$(PYTHON) experiments/benchmark_forward.py --config $(SMOKE_CONFIG)  # Forward solve cost.
	$(MAKE) tables  # Numerical tables built from the smoke summaries.
	$(MAKE) figures  # Figures built from the smoke summaries.
	$(MAKE) online-resource  # Online Resource 1 built from the smoke summaries.

# Build every figure of the manuscript from the tracked summaries.
figures:
	$(PYTHON) figures/make_fig1_model_kernels.py  # Fig. 1, model structure and kernels.
	$(PYTHON) figures/make_fig2_solver_verification.py  # Fig. 2, solver verification.
	$(PYTHON) figures/make_fig3_structural.py  # Fig. 3, structural non-identifiability.
	$(PYTHON) figures/make_fig4_profile_likelihood.py  # Fig. 4, profile likelihoods.
	$(PYTHON) figures/make_fig5_interval_widths.py  # Fig. 5, relative interval widths.
	$(PYTHON) figures/make_fig6_one_vs_two_dimensions.py  # Fig. 6, one against two dimensions.
	$(PYTHON) figures/make_fig7_wolf.py  # Fig. 7, application to wolf space use.

# Build every numerical table of the manuscript from the tracked summaries.
tables:
	$(PYTHON) tables/make_table3_design.py  # Table 3, parameter design.
	$(PYTHON) tables/make_table4_factorial.py  # Table 4, factorial design.
	$(PYTHON) tables/make_table5_convergence.py  # Table 5, solver convergence.
	$(PYTHON) tables/make_table6_identifiability.py  # Table 6, practical identifiability.
	$(PYTHON) tables/make_table7_wolf_data.py  # Table 7, wolf telemetry summary.
	$(PYTHON) tables/make_table8_wolf_posteriors.py  # Table 8, wolf posteriors.

# Build the electronic supplementary material of the manuscript.
online-resource:
	$(PYTHON) tables/make_online_resource_1.py  # Posterior summaries for all 72 conditions.

# Remove generated artefacts that are not tracked by version control.
clean:
	rm -rf results/raw  # Raw experiment output, regenerated by the experiment scripts.
	rm -rf checkpoints  # Sampler checkpoints, regenerated on resume.
	rm -rf .pytest_cache .ruff_cache  # Caches of the test runner and of the linter.
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +  # Byte compiled modules.
