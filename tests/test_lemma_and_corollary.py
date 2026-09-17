# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Verify the injectivity used in Lemma 1 and the sign property of
# Corollary 1 on a fine grid of perceptual ranges.
# Manuscript: Section 3.3, Lemma 1; Section 3.5, Corollary 1.
# Inputs: none. Outputs: assertions of the test runner.

import numpy as np  # Numerical arrays and elementary functions.
import pytest  # Approximate comparison and expected error declaration.

from nmi.kernels import kernel_hat_1d  # Fourier transform of the detection kernel.
from nmi.linear_theory import m_star  # Index of the first nonpositive top-hat multiplier.


# The map of Lemma 1 is injective on the admissible range of angles.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected injectivity.
def test_lemma_1_injectivity():
    angles = np.linspace(1.0e-6, np.pi - 1.0e-6, 20001)  # Angles theta in the open interval.
    cosines = np.cos(angles)  # The cosine, whose injectivity Lemma 1 uses.
    assert np.all(np.diff(cosines) < 0.0)  # The cosine decreases strictly on the interval.


# The two identities of Lemma 1 determine the angle and the constant uniquely.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected uniqueness.
def test_lemma_1_determines_the_pair():
    rng = np.random.default_rng(99)  # Deterministic generator of the inspected pairs.
    for _ in range(500):  # Inspect one random pair at a time.
        angle = rng.uniform(1.0e-3, np.pi - 1.0e-3)  # Angle theta of this pair.
        constant = 10.0 ** rng.uniform(-2.0, 2.0)  # Constant c of this pair.
        first = constant * np.sin(angle)  # First identity of Lemma 1.
        second = constant * np.sin(2.0 * angle)  # Second identity of Lemma 1.
        recovered_angle = float(np.arccos(0.5 * second / first))  # Angle implied by the ratio.
        recovered_constant = first / np.sin(recovered_angle)  # Constant implied by the angle.
        assert recovered_angle == pytest.approx(angle, rel=1.0e-9)  # The angle is recovered.
        assert recovered_constant == pytest.approx(constant, rel=1.0e-9)  # The constant follows.


# The top-hat transform is nonpositive at the index m_star of Corollary 1.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected sign on a fine grid of ranges.
def test_corollary_1_sign():
    radii = np.linspace(1.0e-4, 0.5 - 1.0e-6, 20001)  # Perceptual ranges of the inspected grid.
    for radius in radii:  # Inspect one perceptual range at a time.
        angle = 2.0 * np.pi * float(radius)  # Angle theta of this perceptual range.
        index = m_star(float(radius), 1.0)  # Index m_star of Corollary 1.
        assert float(np.sin(index * angle)) <= 1.0e-12  # The sine is nonpositive at that index.


# The transform at the index m_star is nonpositive for the top-hat kernel.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected sign of the transform itself.
def test_corollary_1_transform_sign():
    for radius in (0.075, 0.15, 0.30, 0.42):  # Perceptual ranges inspected by the test.
        index = m_star(radius, 1.0)  # Index m_star of Corollary 1 at this range.
        value = float(kernel_hat_1d(radius * 2.0 * np.pi * index, "tophat"))  # Transform value.
        assert value <= 1.0e-12  # The top-hat transform is nonpositive at that index.
        assert float(kernel_hat_1d(radius * 2.0 * np.pi * index, "gaussian")) > 0.0  # Gaussian.


# The index m_star matches the values quoted in Table 3 of the manuscript.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected indices.
def test_m_star_matches_table_3():
    assert m_star(0.075, 1.0) == 7  # Index quoted in the first row of Table 3.
    assert m_star(0.15, 1.0) == 4  # Index quoted in the second row of Table 3.
    assert m_star(0.30, 1.0) == 2  # Index quoted in the third row of Table 3.


# A perceptual range outside the admissible interval is rejected.
# Arguments:
#   none.
# Returns:
#   None: the test asserts that the error is raised.
def test_m_star_rejects_inadmissible_ranges():
    with pytest.raises(ValueError, match="0 < R < L / 2"):  # The error states the condition.
        m_star(0.6, 1.0)  # A perceptual range larger than half the domain length.
