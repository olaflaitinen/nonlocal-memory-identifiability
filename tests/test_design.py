# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Verify the experimental design of Section 4.2: the number of
# conditions, the parameter values of Table 3, the index m_star and the
# determinism of the per-condition seeds.
# Manuscript: Section 4.2, Tables 3 and 4.
# Inputs: the tracked design configuration. Outputs: assertions of the runner.

import pytest  # Approximate comparison of floating point values.

from nmi.design import (  # Construction of the experimental design of Section 4.2.
    condition_identifier,  # Stable textual label of one condition.
    design_1d,  # Full factorial design of the one-dimensional experiments.
    initial_density,  # Initial density of the transient conditions.
    table3_values,  # Parameter values of one row of Table 3.
)

# Reference values of Table 3, in the order kernel, radius, kappa_c, kappa, gamma, m_star.
TABLE_3 = [
    ("tophat", 0.075, 1.038, 1.246, 0.01246, 7),  # First row of Table 3.
    ("tophat", 0.15, 1.165, 1.398, 0.01398, 4),  # Second row of Table 3.
    ("tophat", 0.30, 1.982, 2.378, 0.02378, 2),  # Third row of Table 3.
    ("gaussian", 0.075, 1.117, 1.341, 0.01341, None),  # Fourth row of Table 3.
    ("gaussian", 0.15, 1.559, 1.871, 0.01871, None),  # Fifth row of Table 3.
    ("gaussian", 0.30, 5.909, 7.091, 0.07091, None),  # Sixth row of Table 3.
]  # Reference values quoted in the manuscript.


# The factorial design has exactly seventy two transient conditions.
# Arguments:
#   design_config (dict): the design configuration fixture.
# Returns:
#   None: the test asserts the expected counts.
def test_condition_counts(design_config):
    design = design_1d(design_config)  # Factorial design of the one-dimensional experiments.
    assert len(design["transient"]) == 72  # Four noise levels by three designs by three ranges by two kernels.
    assert len(design["stationary"]) == 16  # The sixteen profile likelihood conditions of Table 6.


# Every row of Table 3 is reproduced by the design module.
# Arguments:
#   design_config (dict): the design configuration fixture.
#   row (tuple): one reference row of Table 3.
# Returns:
#   None: the test asserts the expected values.
@pytest.mark.parametrize("row", TABLE_3)  # One test per row of Table 3.
def test_table_3_values(design_config, row):  # Reproduction of the values of Table 3.
    kernel, radius, critical, ratio, advection, index = row  # Reference values of this row.
    values = table3_values(kernel, radius, design_config["design"])  # Computed values of the row.
    assert values["critical_kappa"] == pytest.approx(critical, abs=5.0e-4)  # Critical ratio.
    assert values["aggregation_ratio"] == pytest.approx(ratio, abs=5.0e-4)  # Aggregation ratio.
    assert values["advection"] == pytest.approx(advection, abs=5.0e-6)  # Advection strength.
    assert values["m_star"] == index  # Index m_star of Corollary 1, absent for the Gaussian kernel.


# The condition identifiers follow the documented format.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected identifier.
def test_condition_identifier_format():
    label = condition_identifier("tophat", 0.15, 0.05, "intermediate")  # Identifier of a condition.
    assert label == "k-tophat_R-0.150_eta-0.05_s-intermediate"  # Format given in Section 4.2.


# The per-condition seeds are deterministic and pairwise distinct.
# Arguments:
#   design_config (dict): the design configuration fixture.
# Returns:
#   None: the test asserts the expected determinism.
def test_seeds_are_deterministic(design_config):
    first = design_1d(design_config)["transient"]  # First construction of the factorial design.
    second = design_1d(design_config)["transient"]  # Second construction of the same design.
    assert [item["seed"] for item in first] == [item["seed"] for item in second]  # Determinism.
    assert len({item["seed"] for item in first}) == len(first)  # The seeds are pairwise distinct.
    assert [item["index"] for item in first] == list(range(len(first)))  # Indices are contiguous.


# The initial perturbation is reproducible and scaled to unit maximum amplitude.
# Arguments:
#   design_config (dict): the design configuration fixture.
# Returns:
#   None: the test asserts the expected normalisation.
def test_initial_density_normalisation(design_config):
    design = design_config["design"]  # Block that describes the synthetic design.
    first = initial_density(design, 256)  # First construction of the initial density.
    second = initial_density(design, 256)  # Second construction of the same initial density.
    assert (first == second).all()  # The initial density is reproducible from the recorded seed.
    amplitude = float(design["perturbation_amplitude"])  # Amplitude epsilon of Section 4.2.
    mean_density = float(design["mean_density"])  # Mean density u_bar of the uniform state.
    deviation = (first - mean_density) / (mean_density * amplitude)  # Perturbation phi itself.
    assert float(max(abs(deviation.min()), abs(deviation.max()))) == pytest.approx(1.0)  # Unit.
