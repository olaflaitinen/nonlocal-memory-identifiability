# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Gaussian log-likelihood of the observation model (9) and the forward
# maps that evaluate it for transient and for stationary synthetic data. Every
# parameter is handled in log space, as required by the samplers of Section 4.4.
# Manuscript: Section 4.2, equation (9); Sections 4.3 and 4.4.
# Inputs: parameters in log space, observations and a context mapping.
# Outputs: log-likelihood values and predicted densities.

import numpy as np  # Numerical arrays and elementary functions.

from nmi.design import initial_density, sampling_indices  # Initial data and sampling indices.
from nmi.spectral_1d import PositivityError, simulate_1d  # Forward solver and its error type.
from nmi.steady_state import fixed_point_steady_state  # Fixed-point steady state solver.


# Gaussian log-likelihood of independent observations with a common scale.
# Arguments:
#   observed (numpy.ndarray): the observed values y_ij of equation (9).
#   predicted (numpy.ndarray): the model prediction at the same points.
#   sigma (float): the common standard deviation of the observation noise.
# Returns:
#   float: the log-likelihood, including the normalising constant.
def gaussian_loglik(observed, predicted, sigma):
    residual = np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)  # Residuals.
    count = residual.size  # Number of independent observations entering the likelihood.
    constant = -0.5 * count * np.log(2.0 * np.pi * sigma**2)  # Normalising constant of the density.
    return float(constant - 0.5 * float(np.sum(residual**2)) / sigma**2)  # Gaussian log-likelihood.


# Build the context mapping used by the transient forward map.
# Arguments:
#   condition (dict): one condition of the factorial design.
#   config (dict): the whole configuration mapping.
# Returns:
#   dict: the fixed quantities that the forward map needs at every evaluation.
def transient_context(condition, config):
    design = config["design"]  # Block that describes the synthetic design.
    solver = config["solver"]  # Block that describes the numerical resolutions.
    n_points = int(solver["inference"]["n_points"])  # Grid size of the inference resolution.
    time_step = float(solver["inference"]["time_step"])  # Time step of the inference resolution.
    n_time = int(condition["n_time"])  # Number of observation times of the sampling design.
    t_final = float(design["observation_time"])  # Upper end T of the observation window.
    record_times = np.linspace(0.0, t_final, n_time)  # Equally spaced observation times.
    # Indices of the equally spaced spatial sample points of the design.
    space_index, _ = sampling_indices(int(condition["n_space"]), n_time, n_points, n_time)
    # Assemble the quantities that stay fixed across likelihood evaluations.
    return {
        "kernel": condition["kernel"],  # Detection kernel family of the condition.
        "beta": float(design["beta"]),  # Memory uptake rate beta, fixed to one.
        "domain_length": float(design["domain_length"]),  # Length L of the periodic domain.
        "n_points": n_points,  # Grid size of the inference resolution.
        "time_step": time_step,  # Time step of the inference resolution.
        "t_final": t_final,  # Upper end T of the observation window.
        "record_times": record_times,  # Times at which the density is recorded.
        "space_index": space_index,  # Indices of the retained spatial grid points.
        "initial_density": initial_density(design, n_points),  # Initial density of Section 4.2.
        "initial_map": float(design["initial_map"]),  # Initial cognitive map of Section 4.2.
        "scheme": solver["scheme"],  # Time stepping scheme of Section 4.1.
        "positivity_tolerance": float(solver["positivity_tolerance"]),  # Positivity monitor.
    }


# Predicted densities of the transient forward map at the sampling points.
# Arguments:
#   theta_log (numpy.ndarray): logarithms of (d, gamma, mu, R), in that order.
#   context (dict): the mapping produced by transient_context.
# Returns:
#   numpy.ndarray: predictions of shape (number of times, number of points), or
#   None when the forward solve loses positivity.
def predict_transient(theta_log, context):
    values = np.exp(np.asarray(theta_log, dtype=float))  # Parameters in natural units.
    diffusion, advection, memory_decay, radius = values  # Unpack the four free parameters.
    uptake = context["beta"]  # Memory uptake rate beta of the map equation.
    # Model parameters of equation (1) implied by the proposal.
    params = {
        "diffusion": float(diffusion),  # Diffusion rate d of the density equation.
        "alpha": float(advection / uptake),  # Advection strength alpha implied by gamma.
        "beta": float(uptake),  # Memory uptake rate beta of the map equation.
        "memory_decay": float(memory_decay),  # Memory decay rate mu of the map equation.
        "radius": float(radius),  # Perceptual range R of the detection kernel.
    }
    try:  # A proposal may drive the pseudo-spectral solution negative.
        # Forward solve of model (1) at the inference resolution.
        result = simulate_1d(
            params,  # Model parameters implied by the proposal.
            context["kernel"],  # Detection kernel family of the condition.
            context["initial_density"],  # Initial density of Section 4.2.
            context["initial_map"],  # Initial cognitive map of Section 4.2.
            context["t_final"],  # Upper end T of the observation window.
            context["time_step"],  # Time step of the inference resolution.
            context["n_points"],  # Grid size of the inference resolution.
            context["domain_length"],  # Length L of the periodic domain.
            context["record_times"],  # Times at which the density is recorded.
            context["scheme"],  # Time stepping scheme of Section 4.1.
            context["positivity_tolerance"],  # Relative tolerance of the positivity monitor.
        )  # Forward solution of model (1) at the proposed parameters.
    except PositivityError:  # The proposal is rejected by the positivity monitor.
        return None  # Signal an inadmissible proposal to the caller.
    return result["density"][:, context["space_index"]]  # Predictions at the sampling points.


# Log-likelihood of transient synthetic data at parameters given in log space.
# Arguments:
#   theta_log (numpy.ndarray): logarithms of (d, gamma, mu, R), in that order.
#   data (dict): keys "observations" and "sigma" of the condition.
#   context (dict): the mapping produced by transient_context.
# Returns:
#   float: the log-likelihood, or negative infinity for an inadmissible proposal.
def forward_transient_loglik(theta_log, data, context):
    predicted = predict_transient(theta_log, context)  # Model prediction at the sampling points.
    if predicted is None:  # The forward solve lost positivity at this proposal.
        return -np.inf  # Assign zero likelihood to an inadmissible proposal.
    return gaussian_loglik(data["observations"], predicted, data["sigma"])  # Equation (9).


# Build the context mapping used by the stationary forward map.
# Arguments:
#   condition (dict): one condition of the profile likelihood design.
#   config (dict): the whole configuration mapping.
# Returns:
#   dict: the fixed quantities that the forward map needs at every evaluation.
def stationary_context(condition, config):
    design = config["design"]  # Block that describes the synthetic design.
    solver = config["solver"]  # Block that describes the numerical resolutions.
    n_points = int(solver["inference"]["n_points"])  # Grid size of the inference resolution.
    space_index, _ = sampling_indices(int(condition["n_space"]), 1, n_points, 1)  # Sample points.
    # Assemble the quantities that stay fixed across likelihood evaluations.
    return {
        "kernel": condition["kernel"],  # Detection kernel family of the condition.
        "domain_length": float(design["domain_length"]),  # Length L of the periodic domain.
        "mean_density": float(design["mean_density"]),  # Mean density u_bar of the mass.
        "n_points": n_points,  # Grid size of the inference resolution.
        "space_index": space_index,  # Indices of the retained spatial grid points.
        "damping": float(solver.get("fixed_point_damping", 0.5)),  # Damping of the solver.
        "tolerance": float(solver.get("fixed_point_tolerance", 1.0e-12)),  # Solver tolerance.
        "max_iter": int(solver.get("fixed_point_max_iter", 20000)),  # Solver iteration budget.
    }


# Predicted stationary density at the sampling points.
# Arguments:
#   theta_log (numpy.ndarray): logarithms of (kappa, R), in that order.
#   context (dict): the mapping produced by stationary_context.
# Returns:
#   numpy.ndarray: predictions at the retained spatial grid points, or None when
#   the fixed-point solver fails to converge.
def predict_stationary(theta_log, context):
    values = np.exp(np.asarray(theta_log, dtype=float))  # Parameters in natural units.
    ratio, radius = values  # Aggregation ratio kappa and perceptual range R.
    n_points = context["n_points"]  # Grid size of the inference resolution.
    start = np.full(n_points, context["mean_density"])  # Uniform starting iterate of the solver.
    # Steady state of equation (7) at the proposed aggregation ratio and range.
    result = fixed_point_steady_state(
        float(ratio),  # Aggregation ratio of the proposal.
        float(radius),  # Perceptual range of the proposal.
        context["kernel"],  # Detection kernel family of the condition.
        start,  # Uniform starting iterate of the fixed-point solver.
        context["domain_length"],  # Length L of the periodic domain.
        context["damping"],  # Damping weight of the fixed-point solver.
        context["tolerance"],  # Convergence tolerance of the fixed-point solver.
        context["max_iter"],  # Iteration budget of the fixed-point solver.
    )  # Converged steady state of equation (7) at the proposed parameters.
    if not result["converged"]:  # The fixed-point iteration did not reach the tolerance.
        return None  # Signal an inadmissible proposal to the caller.
    return result["density"][context["space_index"]]  # Predictions at the sampling points.


# Log-likelihood of stationary synthetic data at parameters given in log space.
# Arguments:
#   theta_log (numpy.ndarray): logarithms of (kappa, R), in that order.
#   data (dict): keys "observations" and "sigma" of the condition.
#   context (dict): the mapping produced by stationary_context.
# Returns:
#   float: the log-likelihood, or negative infinity for an inadmissible proposal.
def forward_stationary_loglik(theta_log, data, context):
    predicted = predict_stationary(theta_log, context)  # Model prediction at the sample points.
    if predicted is None:  # The fixed-point solver failed to converge at this proposal.
        return -np.inf  # Assign zero likelihood to an inadmissible proposal.
    return gaussian_loglik(data["observations"], predicted, data["sigma"])  # Equation (9).


# Build a log-posterior callable from a likelihood, a prior and a data set.
# Arguments:
#   loglik (callable): a function of the parameters in log space.
#   prior (nmi.priors.LogUniformPrior): the prior on the same parameters.
# Returns:
#   callable: a function that returns the unnormalised log posterior density.
def make_log_posterior(loglik, prior):
    # Unnormalised log posterior density used by every sampler of the package.
    # Arguments:
    #   theta_log (numpy.ndarray): logarithms of the parameters.
    # Returns:
    #   float: the log prior plus the log-likelihood, or negative infinity.
    def log_posterior(theta_log):
        log_prior = prior.logpdf(theta_log)  # Log density of the log-uniform prior.
        if not np.isfinite(log_prior):  # The proposal lies outside the prior support.
            return -np.inf  # Reject the proposal without a forward solve.
        return log_prior + loglik(theta_log)  # Unnormalised log posterior density.

    return log_posterior  # Callable passed to the samplers of Section 4.4.
