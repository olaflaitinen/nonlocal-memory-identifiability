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


# Number of densely recorded times used to locate the largest density.
DENSE_RECORDS = 101  # Dense record grid from which every sampling design is drawn.


# Generate the reference solution and the noisy data of one condition.
# Arguments:
#   condition (dict): one condition of the factorial design.
#   config (dict): the whole configuration mapping.
#   rng (numpy.random.Generator): generator of the observation noise.
# Returns:
#   dict: the noisy observations, the noise scale, the largest density of the
#   reference simulation, the recorded times and the stationary observations.
def generate_condition_data(condition, config, rng):
    from nmi.design import initial_density, sampling_indices  # Initial data and sample points.
    from nmi.spectral_1d import simulate_1d  # Pseudo-spectral forward solver.
    from nmi.steady_state import fixed_point_steady_state  # Fixed-point steady state solver.

    design = config["design"]  # Block that describes the synthetic design.
    solver = config["solver"]  # Block that describes the numerical resolutions.
    n_points = int(solver["reference"]["n_points"])  # Grid size of the reference resolution.
    time_step = float(solver["reference"]["time_step"])  # Time step of the reference resolution.
    t_final = float(design["observation_time"])  # Upper end T of the observation window.
    dense_times = np.linspace(0.0, t_final, DENSE_RECORDS)  # Dense record grid of the reference.
    # Model parameters of equation (1) for the current condition.
    params = {
        "diffusion": float(condition["diffusion"]),  # Diffusion rate d of the condition.
        "alpha": float(condition["alpha"]),  # Advection strength alpha of the condition.
        "beta": float(condition["beta"]),  # Memory uptake rate beta of the condition.
        "memory_decay": float(condition["memory_decay"]),  # Memory decay rate mu.
        "radius": float(condition["radius"]),  # Perceptual range R of the condition.
    }
    # Reference solve at the reference resolution, which avoids the inverse crime.
    result = simulate_1d(
        params,  # Model parameters of the condition.
        condition["kernel"],  # Detection kernel family of the condition.
        initial_density(design, n_points),  # Initial density of Section 4.2.
        float(design["initial_map"]),  # Initial cognitive map of Section 4.2.
        t_final,  # Upper end T of the observation window.
        time_step,  # Time step of the reference resolution.
        n_points,  # Grid size of the reference resolution.
        float(design["domain_length"]),  # Length L of the periodic domain.
        dense_times,  # Dense record grid from which the design times are drawn.
        solver["scheme"],  # Time stepping scheme of Section 4.1.
        float(solver["positivity_tolerance"]),  # Relative tolerance of the positivity monitor.
    )  # Reference solution of model (1) for this condition.
    u_max = float(np.max(result["density"]))  # Largest density of the reference simulation.
    # Indices of the equally spaced observation points of the sampling design.
    space_index, time_index = sampling_indices(
        int(condition["n_space"]),  # Number of equally spaced spatial points of the design.
        int(condition["n_time"]),  # Number of equally spaced observation times of the design.
        n_points,  # Grid size of the reference resolution.
        DENSE_RECORDS,  # Number of densely recorded times of the reference solution.
    )  # Indices of the observation points of the sampling design.
    clean = sample_observations(result["density"], space_index, time_index)  # Noise free values.
    noisy, sigma = add_gaussian_noise(clean, condition["noise_level"], u_max, rng)  # Equation (9).
    # Stationary state reached by the reference simulation, polished by the fixed-point solver.
    stationary = fixed_point_steady_state(
        float(condition["aggregation_ratio"]),  # Aggregation ratio kappa of the condition.
        float(condition["radius"]),  # Perceptual range R of the condition.
        condition["kernel"],  # Detection kernel family of the condition.
        np.maximum(result["final_density"], 1.0e-12),  # Final transient state as the first iterate.
        float(design["domain_length"]),  # Length L of the periodic domain.
        float(solver.get("fixed_point_damping", 0.5)),  # Damping of the fixed-point solver.
        float(solver.get("fixed_point_tolerance", 1.0e-12)),  # Tolerance of the solver.
        int(solver.get("fixed_point_max_iter", 20000)),  # Iteration budget of the solver.
    )  # Steady state on the branch selected by the transient solution.
    clean_stationary = stationary["density"][space_index]  # Steady state at the sample points.
    # Stationary observations carry the same noise scale as the transient ones.
    noisy_stationary, _ = add_gaussian_noise(clean_stationary, condition["noise_level"], u_max, rng)
    # Assemble the generated data together with the diagnostics of the reference solve.
    return {
        "observations": noisy,  # Noisy transient observations of the condition.
        "clean": clean,  # Noise free transient values at the same points.
        "sigma": sigma,  # Standard deviation of the observation noise.
        "u_max": u_max,  # Largest density of the reference simulation.
        "times": dense_times[time_index],  # Observation times of the sampling design.
        "space_index": space_index,  # Indices of the retained spatial grid points.
        "stationary_observations": noisy_stationary,  # Noisy stationary observations.
        "stationary_clean": clean_stationary,  # Noise free stationary values.
        "stationary_density": stationary["density"],  # Stationary density on the reference grid.
        "stationary_residual": stationary["residual"],  # Residual of the identity of equation (7).
        "dominant_mode": stationary["dominant_mode"],  # Dominant Fourier mode of the steady state.
        "final_density": result["final_density"],  # Final transient state of the reference solve.
        "min_density": result["min_density"],  # Smallest density of the reference solve.
        "mass_error": result["mass_error"],  # Relative mass error of the reference solve.
        "wall_time": result["wall_time"],  # Wall clock cost of the reference solve.
    }
