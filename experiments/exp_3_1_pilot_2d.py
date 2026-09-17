# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Two-dimensional pilot. The script generates the reference solutions
# and the noisy data of the selected two-dimensional conditions, measures the
# cost of one forward solve and fixes the settings of the ensemble sampler.
# Manuscript: Section 5.4, Experiment 3.1; Remark 2.
# Inputs: an experiment configuration. Outputs: one HDF5 archive per condition
# and a tracked summary under results/summary/.

import argparse  # Command line interface of the experiment script.
import sys  # Process exit status of the experiment script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and random number generation.

from nmi.config import load_experiment_config  # Configuration loader with base inheritance.
from nmi.design import table3_values  # Parameter values of Table 3.
from nmi.io import ensure_dir, run_metadata, write_csv, write_hdf5  # Output writers.
from nmi.observation import add_gaussian_noise  # Observation model of equation (9).
from nmi.random_state import generator_from_sequence, spawn_seed_sequences  # Deterministic seeds.
from nmi.spectral_2d import simulate_2d  # Two-dimensional pseudo-spectral solver.

# Number of densely recorded times used to locate the largest density.
DENSE_RECORDS = 21  # Dense record grid of the two-dimensional reference solutions.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Two-dimensional pilot")  # Argument parser.
    parser.add_argument("--config", required=True, help="experiment configuration")  # Configuration.
    parser.add_argument("--condition-index", type=int, default=None, help="one condition")  # Subset.
    parser.add_argument("--output-dir", default=None, help="raw output directory")  # Destination.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# Initial density of the two-dimensional experiments.
# Arguments:
#   n_points (int): number of grid points in each coordinate direction.
#   length (float): side length L of the periodic square.
#   mean_density (float): mean density u_bar of the uniform state.
#   amplitude (float): amplitude epsilon of the initial perturbation.
# Returns:
#   numpy.ndarray: the initial density of shape (n_points, n_points).
def initial_density_2d(n_points, length, mean_density, amplitude):
    from nmi.grids import periodic_grid_2d  # Equispaced grid of the periodic square.

    grid_x, grid_y = periodic_grid_2d(int(n_points), float(length))  # Coordinates of the cells.
    wave = 2.0 * np.pi / float(length)  # Wavenumber of the first mode of the square.
    # Fixed combination of the lowest modes of the square, used in every run.
    shape = np.cos(wave * grid_x) + np.cos(wave * grid_y) + 0.5 * np.cos(wave * (grid_x + grid_y))
    shape = shape / float(np.max(np.abs(shape)))  # Scale to unit maximum absolute value.
    return float(mean_density) * (1.0 + float(amplitude) * shape)  # Perturbed uniform state.


# Build the list of two-dimensional conditions selected for Section 5.4.
# Arguments:
#   config (dict): the configuration mapping of the run.
# Returns:
#   list: the selected conditions, with a stable index and identifier.
def conditions_2d(config):
    settings = config["pilot_2d"]  # Settings of the two-dimensional experiments.
    design = config["design"]  # Block that describes the synthetic design.
    kernels = sorted(design["kernels"])  # Kernel families, in alphabetical order.
    radii = sorted(float(value) for value in settings["radii"])  # Perceptual ranges, ascending.
    noises = sorted(float(value) for value in settings["noise_levels"])  # Noise levels, ascending.
    combinations = []  # Accumulator for the factor combinations in lexicographic order.
    for kernel in kernels:  # Outermost factor of the lexicographic ordering.
        for radius in radii:  # Second factor of the lexicographic ordering.
            for noise in noises:  # Innermost factor of the lexicographic ordering.
                combinations.append((kernel, radius, noise))  # Record the factor combination.
    sequences = spawn_seed_sequences(int(design["seed"]) + 1, len(combinations))  # Per-condition seeds.
    selected = []  # Accumulator for the selected two-dimensional conditions.
    for index, (kernel, radius, noise) in enumerate(combinations):  # Build each condition.
        values = table3_values(kernel, radius, design)  # Parameter values of Table 3.
        record = dict(values)  # Copy so that the Table 3 values are not modified in place.
        record["index"] = index  # Stable integer index of the two-dimensional condition.
        record["identifier"] = f"k-{kernel}_R-{radius:.3f}_eta-{noise:.2f}_2d"  # Stable label.
        record["noise_level"] = float(noise)  # Relative noise level eta of the condition.
        record["diffusion"] = float(design["diffusion"])  # True diffusion rate d.
        record["memory_decay"] = float(design["memory_decay"])  # True memory decay rate mu.
        record["beta"] = float(design["beta"])  # True memory uptake rate beta.
        record["n_space"] = int(settings["n_space"])  # Spatial sample points along each axis.
        record["n_time"] = int(settings["n_time"])  # Number of observation times.
        record["seed"] = int(sequences[index].generate_state(1, dtype=np.uint64)[0])  # Seed.
        selected.append(record)  # Store the completed two-dimensional condition.
    return selected  # The two-dimensional conditions of Section 5.4.


# Entry point of the two-dimensional pilot.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    config = load_experiment_config(arguments.config)  # Merged and validated configuration.
    settings = config["pilot_2d"]  # Settings of the two-dimensional experiments.
    design = config["design"]  # Block that describes the synthetic design.
    solver = config["solver"]  # Block that describes the numerical resolutions.
    length = float(design["domain_length"])  # Side length L of the periodic square.
    t_final = float(design["observation_time"])  # Upper end T of the observation window.
    directory = ensure_dir(arguments.output_dir or Path("results/raw") / str(config["experiment"]))
    conditions = conditions_2d(config)  # Two-dimensional conditions of Section 5.4.
    if arguments.condition_index is not None:  # A single condition was requested.
        conditions = [conditions[int(arguments.condition_index) % len(conditions)]]  # That one.
    records = []  # Accumulator for the rows of the pilot table.
    for condition in conditions:  # Generate the data of each selected condition in turn.
        params = {  # Model parameters of equation (1) for this condition.
            "diffusion": float(condition["diffusion"]),  # Diffusion rate d of the condition.
            "alpha": float(condition["alpha"]),  # Advection strength alpha of the condition.
            "beta": float(condition["beta"]),  # Memory uptake rate beta of the condition.
            "memory_decay": float(condition["memory_decay"]),  # Memory decay rate mu.
            "radius": float(condition["radius"]),  # Perceptual range R of the condition.
        }
        n_reference = int(settings["reference_points"])  # Grid size of the reference solution.
        dense_times = np.linspace(0.0, t_final, DENSE_RECORDS)  # Dense record grid.
        reference = simulate_2d(  # Reference solve at the two-dimensional reference resolution.
            params,  # Model parameters of the condition.
            condition["kernel"],  # Detection kernel family of the condition.
            initial_density_2d(n_reference, length, float(design["mean_density"]), float(design["perturbation_amplitude"])),
            float(design["initial_map"]),  # Initial cognitive map of Section 4.2.
            t_final,  # Upper end T of the observation window.
            float(settings["reference_time_step"]),  # Time step of the reference solution.
            n_reference,  # Grid size of the reference solution.
            length,  # Side length L of the periodic square.
            dense_times,  # Dense record grid from which the design times are drawn.
            solver["scheme"],  # Time stepping scheme of Section 4.1.
            float(solver["positivity_tolerance"]),  # Relative tolerance of the monitor.
        )  # Reference solution of model (1) for this two-dimensional condition.
        u_max = float(np.max(reference["density"]))  # Largest density of the reference solution.
        n_space = int(condition["n_space"])  # Number of spatial sample points along each axis.
        stride = n_reference // n_space  # Stride that selects the equally spaced sample points.
        time_index = np.round(np.linspace(0, DENSE_RECORDS - 1, int(condition["n_time"]))).astype(int)
        clean = reference["density"][np.ix_(time_index)][:, ::stride, ::stride]  # Sampled values.
        generator = generator_from_sequence(np.random.SeedSequence(int(condition["seed"])))  # Seed.
        noisy, sigma = add_gaussian_noise(clean, condition["noise_level"], u_max, generator)
        write_hdf5(  # Archive of the generated two-dimensional data of this condition.
            directory / f"data2d_{condition['index']:03d}.h5",  # Archive of this condition.
            {  # Arrays stored in the archive of this condition.
                "observations": noisy,  # Noisy two-dimensional observations.
                "clean": clean,  # Noise free values at the same points.
                "times": dense_times[time_index],  # Observation times of the sampling design.
            },
            run_metadata(  # Metadata recorded alongside the archive.
                config,  # Configuration mapping of the run.
                {  # Entries that identify the condition in the archive.
                    "condition_index": condition["index"],  # Index of the condition.
                    "identifier": condition["identifier"],  # Label of the condition.
                    "sigma": sigma,  # Standard deviation of the observation noise.
                    "u_max": u_max,  # Largest density of the reference solution.
                    "seed": condition["seed"],  # Deterministic seed of the condition.
                },
            ),
        )  # Archive written for the two-dimensional inference script.
        records.append(  # One row of the two-dimensional pilot table.
            {  # Diagnostics of the reference solve of this condition.
                "condition_index": condition["index"],  # Stable integer index of the condition.
                "identifier": condition["identifier"],  # Stable textual label of the condition.
                "kernel": condition["kernel"],  # Detection kernel family of the condition.
                "radius": condition["radius"],  # Perceptual range R of the condition.
                "noise_level": condition["noise_level"],  # Relative noise level eta.
                "aggregation_ratio": condition["aggregation_ratio"],  # Aggregation ratio kappa.
                "advection": condition["advection"],  # Combined advection strength gamma.
                "sigma": sigma,  # Standard deviation of the observation noise.
                "u_max": u_max,  # Largest density of the reference solution.
                "min_density": reference["min_density"],  # Smallest density of the solve.
                "mass_error": reference["mass_error"],  # Relative mass error of the solve.
                "reference_wall_time": reference["wall_time"],  # Cost of the reference solve.
                "n_walkers": int(settings["n_walkers"]),  # Number of walkers of the sampler.
                "pilot_steps": int(settings["n_steps"]),  # Number of ensemble steps of the pilot.
                "seed": condition["seed"],  # Deterministic seed of the condition.
            }
        )  # Row appended to the pilot table.
        print(f"two-dimensional condition {condition['index']:03d} {condition['identifier']}")
    summary = Path("results/summary") / f"{config['experiment']}_pilot_2d.csv"  # Summary path.
    write_csv(summary, records, list(records[0].keys()))  # Write the tracked summary table.
    print(f"wrote {summary} with {len(records)} row(s)")  # Report the location of the summary.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the experiment and propagate the exit status.
