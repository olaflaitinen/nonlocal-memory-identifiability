# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Weighted log-likelihood (10) of the location fixes of one tracked
# individual under the stationary density predicted by model (1) on the masked
# study area.
# Manuscript: Section 4.5, equation (10).
# Inputs: projected fixes, a stationary density and an effective sample size.
# Outputs: pointwise log densities and the weighted log-likelihood.

import numpy as np  # Numerical arrays and elementary functions.

from nmi.wolf.masked_solver import masked_steady_state  # Stationary density on the masked grid.
from nmi.wolf.study_area import cell_indices  # Mapping from coordinates to grid cells.

# Smallest density assigned to a cell, which avoids an infinite log-likelihood.
DENSITY_FLOOR = 1.0e-300  # Numerical floor applied before taking the logarithm.


# Pointwise log densities of a set of fixes under a density on the masked grid.
# Arguments:
#   x_values (numpy.ndarray): first coordinate of the fixes, in metres.
#   y_values (numpy.ndarray): second coordinate of the fixes, in metres.
#   density (numpy.ndarray): stationary probability density on the grid.
#   area (dict): the mapping produced by build_study_area.
# Returns:
#   numpy.ndarray: the log density evaluated at every fix.
def pointwise_log_density(x_values, y_values, density, area):
    rows, columns = cell_indices(x_values, y_values, area)  # Cells that contain the fixes.
    values = np.asarray(density, dtype=float)[rows, columns]  # Density in the containing cells.
    return np.log(np.maximum(values, DENSITY_FLOOR))  # Log density with a numerical floor.


# Weighted log-likelihood (10) of the fixes of one individual.
# Arguments:
#   log_density (numpy.ndarray): the pointwise log densities of the fixes.
#   n_eff (float): the effective sample size of the autocorrelated fixes.
# Returns:
#   float: the weighted log-likelihood of equation (10).
def weighted_point_loglik(log_density, n_eff):
    values = np.asarray(log_density, dtype=float)  # Pointwise log densities of the fixes.
    count = values.size  # Number n of retained fixes of the individual.
    if count == 0:  # An individual without fixes carries no information.
        return 0.0  # The weighted log-likelihood of an empty sample vanishes.
    return float(n_eff) / float(count) * float(np.sum(values))  # Equation (10) of Section 4.5.


# Pointwise contributions to the weighted log-likelihood (10).
# Arguments:
#   log_density (numpy.ndarray): the pointwise log densities of the fixes.
#   n_eff (float): the effective sample size of the autocorrelated fixes.
# Returns:
#   numpy.ndarray: the weighted contribution of every fix, summing to (10).
def weighted_pointwise(log_density, n_eff):
    values = np.asarray(log_density, dtype=float)  # Pointwise log densities of the fixes.
    count = max(1, values.size)  # Number n of retained fixes, guarded against an empty sample.
    return values * (float(n_eff) / float(count))  # Weighted contribution of every fix.


# Weighted log-likelihood of one individual at given parameters.
# Arguments:
#   theta_log (numpy.ndarray): logarithms of (kappa, R), in that order.
#   individual (dict): the fixes, the mask, the cell width and the sample size.
# Returns:
#   tuple: the weighted log-likelihood and the weighted pointwise contributions.
def individual_loglik(theta_log, individual):
    values = np.exp(np.asarray(theta_log, dtype=float))  # Parameters in natural units.
    ratio, radius = float(values[0]), float(values[1])  # Aggregation ratio and perceptual range.
    area = individual["area"]  # Study area mapping produced by build_study_area.
    # Stationary density implied by the proposed aggregation ratio and range.
    result = masked_steady_state(
        ratio,  # Aggregation ratio kappa of the proposal.
        radius,  # Perceptual range R of the proposal.
        individual["kernel"],  # Detection kernel family of the fitted model.
        area["mask"],  # Boolean mask of the study area.
        area["cell"],  # Uniform cell width of the Cartesian grid.
        individual.get("initial_density"),  # Utilisation distribution used as the first iterate.
        individual.get("damping", 0.5),  # Relaxation weight of the fixed-point solver.
        individual.get("tolerance", 1.0e-10),  # Convergence tolerance of the solver.
        individual.get("max_iter", 5000),  # Iteration budget of the fixed-point solver.
    )  # Converged stationary density of Proposition 4 at the proposal.
    if not result["converged"]:  # The fixed-point solver failed to reach the tolerance.
        return -np.inf, None  # Assign zero likelihood to an inadmissible proposal.
    # Log density of the fitted stationary model at every retained fix.
    log_density = pointwise_log_density(
        individual["easting"],  # First coordinate of the retained fixes.
        individual["northing"],  # Second coordinate of the retained fixes.
        result["density"],  # Stationary density on the masked grid.
        area,  # Study area mapping that defines the grid.
    )  # Log density of the stationary model at every retained fix.
    pointwise = weighted_pointwise(log_density, individual["n_eff"])  # Weighted contributions.
    return float(np.sum(pointwise)), pointwise  # Weighted log-likelihood and its contributions.


# Weighted log-likelihood of the model without nonlocal advection.
# The stationary density of that model is uniform on the study area.
# Arguments:
#   individual (dict): the fixes, the mask, the cell width and the sample size.
# Returns:
#   tuple: the weighted log-likelihood and the weighted pointwise contributions.
def uniform_loglik(individual):
    area = individual["area"]  # Study area mapping produced by build_study_area.
    mask = area["mask"]  # Boolean mask of the study area.
    cell = area["cell"]  # Uniform cell width of the Cartesian grid.
    total = float(np.sum(mask)) * cell * cell  # Area of the study region in square metres.
    density = mask.astype(float) / total  # Uniform probability density on the study area.
    # Log density of the fitted stationary model at every retained fix.
    log_density = pointwise_log_density(
        individual["easting"],  # First coordinate of the retained fixes.
        individual["northing"],  # Second coordinate of the retained fixes.
        density,  # Uniform density of the model without nonlocal advection.
        area,  # Study area mapping that defines the grid.
    )  # Log density of the uniform model at every retained fix.
    pointwise = weighted_pointwise(log_density, individual["n_eff"])  # Weighted contributions.
    return float(np.sum(pointwise)), pointwise  # Weighted log-likelihood and its contributions.
