# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Construction of the synthetic experimental design of the manuscript.
# The module builds the 72 transient conditions of the full factorial design,
# the 16 profile likelihood conditions analysed with stationary and transient
# data, the parameter values of Table 3 and the deterministic per-condition
# seeds.
# Manuscript: Section 4.2, Tables 3 and 4.
# Inputs: a configuration mapping. Outputs: lists of condition dictionaries and
# the Table 3 parameter values.

import numpy as np  # Numerical arrays and random number generation.

from nmi.grids import periodic_grid_1d  # Equispaced grid of the periodic domain.
from nmi.linear_theory import m_star  # Index of the first nonpositive top-hat multiplier.
from nmi.random_state import sequence_label, spawn_seed_sequences  # Deterministic seeding.
from nmi.steady_state import critical_kappa  # Onset value of the aggregation ratio.


# Parameter values of Table 3 for one kernel family and perceptual range.
# Arguments:
#   kernel (str): "tophat" or "gaussian".
#   radius (float): perceptual range R of the detection kernel.
#   design (dict): the design block of the configuration file.
# Returns:
#   dict: the critical ratio, the aggregation ratio, the combined advection
#   strength, the advection strength and the index m_star where it applies.
def table3_values(kernel, radius, design):
    domain_length = float(design["domain_length"])  # Length L of the periodic domain.
    mean_density = float(design["mean_density"])  # Mean density u_bar of the uniform state.
    diffusion = float(design["diffusion"])  # Diffusion rate d of the density equation.
    memory_decay = float(design["memory_decay"])  # Memory decay rate mu of the map equation.
    uptake = float(design["beta"])  # Memory uptake rate beta, fixed to one by Proposition 1.
    factor = float(design["supercriticality"])  # Factor by which kappa exceeds the onset value.
    critical = critical_kappa(radius, kernel, domain_length, mean_density)  # Onset ratio.
    ratio = factor * critical  # Aggregation ratio of equation (8) for this condition.
    advection = ratio * diffusion * memory_decay  # Combined advection strength gamma.
    index = m_star(radius, domain_length) if kernel == "tophat" else None  # Corollary 1 index.
    # Assemble one row of Table 3 together with the derived parameters.
    return {
        "kernel": kernel,  # Detection kernel family of this row of Table 3.
        "radius": float(radius),  # Perceptual range R of this row of Table 3.
        "critical_kappa": float(critical),  # Critical ratio kappa_c(R) of Proposition 3.
        "aggregation_ratio": float(ratio),  # Aggregation ratio kappa of equation (8).
        "advection": float(advection),  # Combined advection strength gamma = kappa d mu.
        "alpha": float(advection / uptake),  # Advection strength alpha, equal to gamma when beta is one.
        "m_star": index,  # Index m_star of Corollary 1, absent for the Gaussian kernel.
    }


# Initial perturbation of the density, fixed by the seed of the configuration.
# Arguments:
#   design (dict): the design block of the configuration file.
#   n_points (int): number of grid points of the discretisation.
# Returns:
#   numpy.ndarray: the perturbation phi, scaled to unit maximum absolute value.
def initial_perturbation(design, n_points):
    domain_length = float(design["domain_length"])  # Length L of the periodic domain.
    n_modes = int(design.get("perturbation_modes", 8))  # Number of modes in the perturbation.
    seed = int(design["perturbation_seed"])  # Seed that fixes the perturbation once and for all.
    rng = np.random.default_rng(seed)  # Generator dedicated to the initial perturbation.
    amplitudes = rng.standard_normal(n_modes)  # Standard normal amplitudes z_m of the modes.
    phases = rng.uniform(0.0, 2.0 * np.pi, n_modes)  # Uniform phases psi_m of the modes.
    grid = periodic_grid_1d(n_points, domain_length)  # Grid points of the periodic domain.
    field = np.zeros(n_points)  # Accumulator for the sum over the retained modes.
    for index in range(n_modes):  # Add the contribution of each mode m = 1, ..., n_modes.
        mode = index + 1  # Mode index m of the current term of the sum.
        argument = 2.0 * np.pi * mode * grid / domain_length + phases[index]  # Phase of the mode.
        field = field + amplitudes[index] * np.cos(argument) / mode  # Contribution of the mode.
    return field / float(np.max(np.abs(field)))  # Scale to unit maximum absolute value.


# Initial density of the transient experiments.
# Arguments:
#   design (dict): the design block of the configuration file.
#   n_points (int): number of grid points of the discretisation.
# Returns:
#   numpy.ndarray: the initial density u_0 = u_bar (1 + epsilon phi).
def initial_density(design, n_points):
    mean_density = float(design["mean_density"])  # Mean density u_bar of the uniform state.
    amplitude = float(design["perturbation_amplitude"])  # Perturbation amplitude epsilon.
    perturbation = initial_perturbation(design, n_points)  # Fixed random perturbation phi.
    return mean_density * (1.0 + amplitude * perturbation)  # Initial density of Section 4.2.


# Identifier string of one condition of the factorial design.
# Arguments:
#   kernel (str): detection kernel family.
#   radius (float): perceptual range R.
#   noise (float): relative noise level eta.
#   sampling (str): name of the sampling design.
# Returns:
#   str: the stable identifier of the condition, for example
#   "k-tophat_R-0.150_eta-0.05_s-intermediate".
def condition_identifier(kernel, radius, noise, sampling):
    return f"k-{kernel}_R-{radius:.3f}_eta-{noise:.2f}_s-{sampling}"  # Stable textual label.


# Full factorial design of the one-dimensional synthetic experiments.
# The conditions are enumerated in lexicographic order of the tuple
# (kernel, radius, noise level, sampling design), with numeric factors in
# ascending order and textual factors in alphabetical order, which fixes the
# integer index of every condition once and for all.
# Arguments:
#   config (dict): the whole configuration mapping.
# Returns:
#   dict: the key "transient" holds the 72 transient conditions and the key
#   "stationary" holds the 16 profile likelihood conditions of Table 6.
def design_1d(config):
    design = config["design"]  # Block that describes the synthetic design.
    kernels = sorted(design["kernels"])  # Kernel families, in alphabetical order.
    radii = sorted(float(value) for value in design["perceptual_ranges"])  # Ranges, ascending.
    noises = sorted(float(value) for value in design["noise_levels"])  # Noise levels, ascending.
    samplings = sorted(design["sampling_designs"])  # Sampling designs, in alphabetical order.
    combinations = []  # Accumulator for the factor combinations in lexicographic order.
    for kernel in kernels:  # Outermost factor of the lexicographic ordering.
        for radius in radii:  # Second factor of the lexicographic ordering.
            for noise in noises:  # Third factor of the lexicographic ordering.
                for sampling in samplings:  # Innermost factor of the lexicographic ordering.
                    combinations.append((kernel, radius, noise, sampling))  # Record the tuple.
    sequences = spawn_seed_sequences(int(design["seed"]), len(combinations))  # Per-condition seeds.
    transient = []  # Accumulator for the transient conditions of the design.
    for index, (kernel, radius, noise, sampling) in enumerate(combinations):  # Build each condition.
        values = table3_values(kernel, radius, design)  # Parameter values of Table 3.
        record = dict(values)  # Copy so that the Table 3 values are not modified in place.
        record["index"] = index  # Stable integer index of the condition.
        record["identifier"] = condition_identifier(kernel, radius, noise, sampling)  # Label.
        record["noise_level"] = float(noise)  # Relative noise level eta of the condition.
        record["sampling"] = sampling  # Name of the sampling design of the condition.
        record["n_space"] = int(design["sampling_designs"][sampling]["n_space"])  # Spatial points.
        record["n_time"] = int(design["sampling_designs"][sampling]["n_time"])  # Time points.
        record["diffusion"] = float(design["diffusion"])  # True diffusion rate d.
        record["memory_decay"] = float(design["memory_decay"])  # True memory decay rate mu.
        record["beta"] = float(design["beta"])  # True memory uptake rate beta.
        record["data_type"] = "transient"  # Every condition of the factorial design is transient.
        record["seed"] = sequence_label(sequences[index])  # Deterministic seed of the condition.
        transient.append(record)  # Store the completed condition.
    # Both groups of conditions are returned together for the experiment scripts.
    return {"transient": transient, "stationary": profile_conditions(config, transient)}


# The 16 profile likelihood conditions reported in Table 6 of the manuscript.
# Section 4.2 restricts them to the intermediate sampling design and to the
# perceptual range 0.15, with both kernel families and all four noise levels.
# Each of those eight settings is analysed twice, once with stationary data and
# once with transient data for comparison, which gives the 16 conditions.
# Arguments:
#   config (dict): the whole configuration mapping.
#   transient (list): the transient conditions of the factorial design.
# Returns:
#   list: the 16 profile likelihood conditions, in a stable order.
def profile_conditions(config, transient):
    design = config["design"]  # Block that describes the synthetic design.
    target_radius = float(design["profile_radius"])  # Perceptual range of the profile conditions.
    target_sampling = design["profile_sampling"]  # Sampling design of the profile conditions.
    selected = []  # Accumulator for the profile likelihood conditions.
    for data_type in ("stationary", "transient"):  # Both data types enter Table 6.
        for record in transient:  # Scan the factorial design for the matching settings.
            matches_radius = abs(record["radius"] - target_radius) < 1.0e-12  # Radius matches.
            matches_sampling = record["sampling"] == target_sampling  # Sampling design matches.
            if not (matches_radius and matches_sampling):  # The setting is not part of Table 6.
                continue  # Skip conditions that Section 4.2 excludes from the profiles.
            entry = dict(record)  # Copy so that the factorial condition stays unchanged.
            entry["data_type"] = data_type  # Data type analysed by this profile condition.
            entry["profile_index"] = len(selected)  # Stable integer index within Table 6.
            entry["identifier"] = f"{record['identifier']}_d-{data_type}"  # Label with the data type.
            selected.append(entry)  # Store the completed profile likelihood condition.
    return selected  # The 16 profile likelihood conditions of Table 6.


# Indices of the spatial and temporal observation points of a sampling design.
# Arguments:
#   n_space (int): number of equally spaced spatial points on the torus.
#   n_time (int): number of equally spaced observation times, including t = 0.
#   n_points (int): number of grid points of the reference discretisation.
#   n_records (int): number of recorded times of the reference simulation.
# Returns:
#   tuple: integer index arrays into the spatial grid and into the record axis.
def sampling_indices(n_space, n_time, n_points, n_records):
    if n_points % n_space != 0:  # Equally spaced sampling requires a divisor of the grid size.
        # Unequal spacing would break the observation model of equation (9).
        raise ValueError(f"The grid size {n_points} is not a multiple of {n_space} sample points")
    space = np.arange(n_space) * (n_points // n_space)  # Equally spaced points on the torus.
    time = np.round(np.linspace(0, n_records - 1, n_time)).astype(int)  # Equally spaced records.
    return space, time  # Index arrays used by the observation model of equation (9).
