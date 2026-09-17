# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Verify the log-uniform priors and the Gaussian log-likelihood of the
# observation model (9) against direct formulas.
# Manuscript: Section 4.2, equation (9); Section 4.4, prior bounds of Table 3.
# Inputs: none. Outputs: assertions of the test runner.

import numpy as np  # Numerical arrays and random number generation.
import pytest  # Approximate comparison of floating point values.

from nmi.likelihood import gaussian_loglik  # Gaussian log-likelihood of equation (9).
from nmi.observation import add_gaussian_noise, noise_scale, sample_observations  # Observations.
from nmi.priors import LogUniformPrior, stationary_prior, transient_prior  # Priors of Table 3.


# The log-uniform prior integrates to one over its support.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected normalisation.
def test_prior_is_normalised():
    prior = LogUniformPrior(("a", "b"), [0.01, 0.5], [1.0, 5.0])  # Prior on two parameters.
    grid_a = np.linspace(np.log(0.01), np.log(1.0), 4001)  # Grid of the first parameter.
    grid_b = np.linspace(np.log(0.5), np.log(5.0), 4001)  # Grid of the second parameter.
    density = float(np.exp(prior.logpdf(np.array([grid_a[10], grid_b[10]]))))  # Constant density.
    area = (grid_a[-1] - grid_a[0]) * (grid_b[-1] - grid_b[0])  # Area of the prior support.
    assert density * area == pytest.approx(1.0)  # The prior integrates to one over its support.


# The prior vanishes outside its support and is finite inside it.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected support.
def test_prior_support():
    prior = LogUniformPrior(("a",), [0.1], [10.0])  # Prior on a single positive parameter.
    assert np.isfinite(prior.logpdf(np.array([np.log(1.0)])))  # Inside the prior support.
    assert prior.logpdf(np.array([np.log(0.01)])) == -np.inf  # Below the lower bound.
    assert prior.logpdf(np.array([np.log(100.0)])) == -np.inf  # Above the upper bound.
    assert prior.bounds() == [(pytest.approx(np.log(0.1)), pytest.approx(np.log(10.0)))]  # Bounds.


# Samples of the prior lie inside its support.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected range of the samples.
def test_prior_samples_lie_in_the_support():
    prior = LogUniformPrior(("a", "b"), [0.01, 0.5], [1.0, 5.0])  # Prior on two parameters.
    draws = prior.sample(np.random.default_rng(7), size=500)  # Draws from the prior in log space.
    assert draws.shape == (500, 2)  # One row per draw and one column per parameter.
    assert bool(np.all(draws >= prior.log_lower))  # Every draw lies above the lower bounds.
    assert bool(np.all(draws <= prior.log_upper))  # Every draw lies below the upper bounds.


# Inadmissible prior bounds are rejected with an informative error.
# Arguments:
#   none.
# Returns:
#   None: the test asserts that the errors are raised.
def test_prior_rejects_inadmissible_bounds():
    with pytest.raises(ValueError, match="strictly positive"):  # A nonpositive lower bound.
        LogUniformPrior(("a",), [0.0], [1.0])  # Zero is outside the admissible range.
    with pytest.raises(ValueError, match="must exceed"):  # An inverted interval.
        LogUniformPrior(("a",), [2.0], [1.0])  # The upper bound is below the lower bound.


# The priors of Table 3 carry the documented bounds.
# Arguments:
#   design_config (dict): the design configuration fixture.
# Returns:
#   None: the test asserts the expected bounds.
def test_table_3_priors(design_config):
    bounds = design_config["design"]["priors"]  # Prior bounds given with Table 3.
    prior = transient_prior(0.01398, bounds)  # Prior of a condition with the reference gamma.
    natural = prior.natural_bounds()  # Prior bounds in natural units.
    assert natural[0] == (pytest.approx(0.001), pytest.approx(0.1))  # Bounds of the diffusion rate.
    assert natural[1] == (pytest.approx(0.001398), pytest.approx(0.1398))  # Bounds of gamma.
    assert natural[2] == (pytest.approx(0.1), pytest.approx(10.0))  # Bounds of the decay rate.
    assert natural[3] == (pytest.approx(0.01), pytest.approx(0.49))  # Bounds of the range R.
    stationary = stationary_prior(1.398, bounds)  # Prior of the stationary conditions.
    assert stationary.names == ("aggregation_ratio", "radius")  # Parameters of Theorem 1.


# The Gaussian log-likelihood agrees with a direct evaluation of the formula.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected agreement.
def test_gaussian_loglik_matches_direct_formula():
    rng = np.random.default_rng(11)  # Deterministic generator of the compared values.
    observed = rng.standard_normal((5, 7))  # Observed values of the comparison.
    predicted = rng.standard_normal((5, 7))  # Predicted values of the comparison.
    sigma = 0.37  # Standard deviation of the observation noise.
    residual = observed - predicted  # Residuals entering the log-likelihood.
    expected = float(  # Direct evaluation of the Gaussian log density.
        -0.5 * residual.size * np.log(2.0 * np.pi * sigma**2)  # Normalising constant.
        - 0.5 * float(np.sum(residual**2)) / sigma**2  # Quadratic term of the log density.
    )  # Expected value of the log-likelihood.
    assert gaussian_loglik(observed, predicted, sigma) == pytest.approx(expected)  # Agreement.


# The observation model samples and perturbs the recorded solution as specified.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected sampling and noise scale.
def test_observation_model():
    record = np.arange(60.0).reshape(6, 10)  # Recorded densities of an artificial solution.
    sampled = sample_observations(record, np.array([0, 5]), np.array([0, 3]))  # Sampled values.
    assert sampled.shape == (2, 2)  # Two observation times by two spatial sample points.
    assert sampled[1, 1] == pytest.approx(35.0)  # Value at the fourth record and sixth point.
    assert noise_scale(0.05, 4.0) == pytest.approx(0.2)  # Noise scale sigma of equation (9).
    noisy, sigma = add_gaussian_noise(sampled, 0.05, 4.0, np.random.default_rng(1))  # Noisy data.
    assert sigma == pytest.approx(0.2)  # The reported noise scale matches equation (9).
    assert noisy.shape == sampled.shape  # The noise does not change the shape of the data.
