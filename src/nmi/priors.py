# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Independent log-uniform prior distributions on the model parameters,
# as specified with Table 3 of the manuscript.
# Manuscript: Section 4.4, Bayesian inference; prior bounds given with Table 3.
# Inputs: parameter bounds. Outputs: log densities, samples and bounds.

import numpy as np  # Numerical arrays and random number generation.


# Independent log-uniform prior on a vector of strictly positive parameters.
# The class stores the bounds in natural units and evaluates the density of the
# logarithms, because every sampler of this package works in log space.
class LogUniformPrior:
    # Construct the prior from the lower and upper bounds of each parameter.
    # Arguments:
    #   names (sequence): parameter names, in the order used by the samplers.
    #   lower (sequence): lower bounds in natural units, strictly positive.
    #   upper (sequence): upper bounds in natural units, strictly positive.
    # Returns:
    #   None: the constructor stores the validated bounds on the instance.
    def __init__(self, names, lower, upper):
        self.names = tuple(names)  # Parameter names in the order used by the samplers.
        self.lower = np.asarray(lower, dtype=float)  # Lower bounds in natural units.
        self.upper = np.asarray(upper, dtype=float)  # Upper bounds in natural units.
        if np.any(self.lower <= 0.0) or np.any(self.upper <= 0.0):  # Log-uniform needs positivity.
            raise ValueError("Log-uniform prior bounds must be strictly positive")  # Reject.
        if np.any(self.upper <= self.lower):  # Each interval must have a positive length.
            # An empty or inverted interval carries no probability mass.
            raise ValueError("Log-uniform prior upper bounds must exceed the lower bounds")
        self.log_lower = np.log(self.lower)  # Lower bounds in log space.
        self.log_upper = np.log(self.upper)  # Upper bounds in log space.
        self.log_width = self.log_upper - self.log_lower  # Width of each interval in log space.

    # Log density of the prior evaluated at a point given in log space.
    # Arguments:
    #   log_theta (numpy.ndarray): logarithms of the parameters.
    # Returns:
    #   float: the log density, or negative infinity outside the support.
    def logpdf(self, log_theta):
        point = np.asarray(log_theta, dtype=float)  # Parameters in log space.
        if np.any(point < self.log_lower) or np.any(point > self.log_upper):  # Outside the support.
            return -np.inf  # The prior density vanishes outside its support.
        return float(-np.sum(np.log(self.log_width)))  # Constant density of the log-uniform law.

    # Draw independent samples from the prior in log space.
    # Arguments:
    #   rng (numpy.random.Generator): the generator used for the draw.
    #   size (int): number of samples requested.
    # Returns:
    #   numpy.ndarray: an array of shape (size, number of parameters).
    def sample(self, rng, size=1):
        draws = rng.uniform(size=(int(size), self.log_lower.size))  # Uniform draws on the unit cube.
        return self.log_lower + draws * self.log_width  # Affine map onto the prior support.

    # Bounds of the prior support in log space.
    # Arguments:
    #   none.
    # Returns:
    #   list: one tuple of lower and upper bound per parameter, in log space.
    def bounds(self):
        # One pair of bounds per parameter, in the order used by the samplers.
        return [(float(low), float(high)) for low, high in zip(self.log_lower, self.log_upper, strict=True)]

    # Bounds of the prior support in natural units.
    # Arguments:
    #   none.
    # Returns:
    #   list: one tuple of lower and upper bound per parameter, in natural units.
    def natural_bounds(self):
        # One pair of bounds per parameter, in the order used by the samplers.
        return [(float(low), float(high)) for low, high in zip(self.lower, self.upper, strict=True)]


# Build the prior of the one-dimensional synthetic experiments.
# Arguments:
#   gamma_true (float): the true combined advection strength of the condition.
#   bounds (dict): prior bounds read from the configuration file.
# Returns:
#   LogUniformPrior: the prior on (d, gamma, mu, R) in that order.
def transient_prior(gamma_true, bounds):
    names = ("diffusion", "advection", "memory_decay", "radius")  # Order used by the samplers.
    # Lower bounds of the four parameters, in the order given by names.
    lower = [
        float(bounds["diffusion"][0]),  # Lower bound of the diffusion rate d.
        float(gamma_true) / float(bounds["advection_factor"]),  # Lower bound of gamma.
        float(bounds["memory_decay"][0]),  # Lower bound of the memory decay rate mu.
        float(bounds["radius"][0]),  # Lower bound of the perceptual range R.
    ]
    # Upper bounds of the four parameters, in the order given by names.
    upper = [
        float(bounds["diffusion"][1]),  # Upper bound of the diffusion rate d.
        float(gamma_true) * float(bounds["advection_factor"]),  # Upper bound of gamma.
        float(bounds["memory_decay"][1]),  # Upper bound of the memory decay rate mu.
        float(bounds["radius"][1]),  # Upper bound of the perceptual range R.
    ]
    return LogUniformPrior(names, lower, upper)  # Independent log-uniform prior on the four parameters.


# Build the prior of the stationary experiments on the aggregation ratio.
# Arguments:
#   kappa_true (float): the true aggregation ratio of the condition.
#   bounds (dict): prior bounds read from the configuration file.
# Returns:
#   LogUniformPrior: the prior on (kappa, R) in that order.
def stationary_prior(kappa_true, bounds):
    names = ("aggregation_ratio", "radius")  # Order used by the stationary samplers.
    factor = float(bounds["advection_factor"])  # Same multiplicative width as for gamma.
    lower = [float(kappa_true) / factor, float(bounds["radius"][0])]  # Lower bounds of the pair.
    upper = [float(kappa_true) * factor, float(bounds["radius"][1])]  # Upper bounds of the pair.
    return LogUniformPrior(names, lower, upper)  # Independent log-uniform prior on the pair.
