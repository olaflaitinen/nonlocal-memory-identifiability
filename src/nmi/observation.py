# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Observation model (9) of the manuscript. The module samples a
# recorded solution at the spatial and temporal points of a sampling design and
# adds independent Gaussian noise whose scale is a fixed fraction of the
# largest density of the reference simulation.
# Manuscript: Section 4.2, equation (9).
# Inputs: recorded densities and sampling indices. Outputs: sampled and noisy
# observations as NumPy arrays.

import numpy as np  # Numerical arrays and random number generation.


# Sample a recorded solution at the points of a sampling design.
# Arguments:
#   u_record (numpy.ndarray): recorded densities of shape (n_records, n_points).
#   x_idx (numpy.ndarray): indices of the retained spatial grid points.
#   t_idx (numpy.ndarray): indices of the retained record times.
# Returns:
#   numpy.ndarray: the sampled densities of shape (len(t_idx), len(x_idx)).
def sample_observations(u_record, x_idx, t_idx):
    records = np.asarray(u_record, dtype=float)  # Recorded densities of the reference solution.
    return records[np.ix_(np.asarray(t_idx, dtype=int), np.asarray(x_idx, dtype=int))]  # Subgrid.


# Noise standard deviation of the observation model (9).
# Arguments:
#   noise_level (float): the relative noise level eta.
#   u_max (float): the largest density of the reference simulation.
# Returns:
#   float: the standard deviation sigma = eta max u.
def noise_scale(noise_level, u_max):
    return float(noise_level) * float(u_max)  # Absolute noise scale of equation (9).


# Add independent Gaussian noise to sampled densities.
# Arguments:
#   y (numpy.ndarray): the noise free sampled densities.
#   noise_level (float): the relative noise level eta.
#   u_max (float): the largest density of the reference simulation.
#   rng (numpy.random.Generator): the generator used for the draw.
# Returns:
#   tuple: the noisy observations and the noise standard deviation sigma.
def add_gaussian_noise(y, noise_level, u_max, rng):
    clean = np.asarray(y, dtype=float)  # Noise free sampled densities of the condition.
    sigma = noise_scale(noise_level, u_max)  # Absolute noise scale of equation (9).
    noise = rng.standard_normal(clean.shape) * sigma  # Independent standard normal draws.
    return clean + noise, sigma  # Noisy observations together with the noise scale.
