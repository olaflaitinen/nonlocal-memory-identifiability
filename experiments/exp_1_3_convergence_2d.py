# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Convergence study of the two-dimensional pseudo-spectral solver and
# measurement of the cost of one two-dimensional forward solve, which fixes the
# resolution used by the two-dimensional experiments of Section 5.4.
# Manuscript: Section 5.1, Experiment 1.3; Remark 2.
# Inputs: an experiment configuration. Outputs: a tracked summary under
# results/summary/.

import argparse  # Command line interface of the experiment script.
import sys  # Process exit status of the experiment script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and elementary functions.

from nmi.config import load_experiment_config  # Configuration loader with base inheritance.
from nmi.design import table3_values  # Parameter values of Table 3.
from nmi.grids import periodic_grid_2d  # Equispaced grid of the periodic square.
from nmi.io import write_csv  # Writer of the tracked summary table.
from nmi.spectral_2d import simulate_2d  # Two-dimensional pseudo-spectral solver.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Two-dimensional convergence")  # Argument parser.
    parser.add_argument("--config", required=True, help="experiment configuration")  # Configuration.
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
    grid_x, grid_y = periodic_grid_2d(int(n_points), float(length))  # Coordinates of the cells.
    wave = 2.0 * np.pi / float(length)  # Wavenumber of the first mode of the square.
    # Fixed combination of the lowest modes of the square, used in every run.
    shape = np.cos(wave * grid_x) + np.cos(wave * grid_y) + 0.5 * np.cos(wave * (grid_x + grid_y))
    shape = shape / float(np.max(np.abs(shape)))  # Scale to unit maximum absolute value.
    return float(mean_density) * (1.0 + float(amplitude) * shape)  # Perturbed uniform state.


# Restrict a two-dimensional solution to a coarser subgrid.
# Arguments:
#   values (numpy.ndarray): the solution on the fine grid.
#   n_coarse (int): number of points of the coarse grid in each direction.
# Returns:
#   numpy.ndarray: the solution restricted to the coarse grid points.
def restrict_to_grid_2d(values, n_coarse):
    n_fine = values.shape[-1]  # Number of points of the fine grid in each direction.
    if n_fine % n_coarse != 0:  # The coarse grid must be a subgrid of the fine grid.
        raise ValueError(f"The grid of {n_fine} points is not a multiple of {n_coarse}")  # Reject.
    stride = n_fine // n_coarse  # Stride that maps the fine grid onto the coarse grid.
    return values[..., ::stride, ::stride]  # Subsample the fine solution onto the coarse grid.


# Entry point of the two-dimensional convergence study.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    config = load_experiment_config(arguments.config)  # Merged and validated configuration.
    settings = config["convergence_2d"]  # Settings of the two-dimensional convergence study.
    design = config["design"]  # Block that describes the synthetic design.
    solver = config["solver"]  # Block that describes the numerical resolutions.
    length = float(design["domain_length"])  # Side length L of the periodic square.
    mean_density = float(design["mean_density"])  # Mean density u_bar of the uniform state.
    amplitude = float(settings["perturbation_amplitude"])  # Amplitude of the perturbation.
    radius = float(settings["radius"])  # Perceptual range of the convergence study.
    evaluation_time = float(settings["evaluation_time"])  # Time at which the error is measured.
    records = []  # Accumulator for the rows of the convergence table.
    for kernel in settings["kernels"]:  # Repeat the study for each detection kernel family.
        values = table3_values(kernel, radius, design)  # Parameter values of Table 3.
        params = {  # Model parameters of equation (1) for this kernel family.
            "diffusion": float(design["diffusion"]),  # Diffusion rate d of the condition.
            "alpha": float(values["alpha"]),  # Advection strength alpha of the condition.
            "beta": float(design["beta"]),  # Memory uptake rate beta of the condition.
            "memory_decay": float(design["memory_decay"]),  # Memory decay rate mu.
            "radius": radius,  # Perceptual range R of the condition.
        }
        n_reference = int(settings["reference_grid"])  # Grid size of the reference solution.
        reference = simulate_2d(  # Fine reference solve used as the ground truth.
            params,  # Model parameters of the condition.
            kernel,  # Detection kernel family of the condition.
            initial_density_2d(n_reference, length, mean_density, amplitude),  # Initial density.
            float(design["initial_map"]),  # Initial cognitive map of Section 4.2.
            evaluation_time,  # End of the reference solve.
            float(settings["reference_time_step"]),  # Time step of the reference solve.
            n_reference,  # Grid size of the reference solve.
            length,  # Side length L of the periodic square.
            [evaluation_time],  # A single record at the evaluation time.
            solver["scheme"],  # Time stepping scheme of Section 4.1.
            float(solver["positivity_tolerance"]),  # Relative tolerance of the monitor.
        )  # Reference solution against which every error is measured.
        reference_state = reference["density"][0]  # Reference state at the evaluation time.
        previous_error = None  # Error of the previous grid, used for the observed order.
        for n_points in settings["grids"]:  # Refine the grid at the fixed time step.
            result = simulate_2d(  # Forward solve on the current grid.
                params,  # Model parameters of the condition.
                kernel,  # Detection kernel family of the condition.
                initial_density_2d(int(n_points), length, mean_density, amplitude),  # Initial data.
                float(design["initial_map"]),  # Initial cognitive map of Section 4.2.
                evaluation_time,  # End of the forward solve.
                float(settings["time_step"]),  # Time step of the forward solve.
                int(n_points),  # Grid size of the forward solve.
                length,  # Side length L of the periodic square.
                [evaluation_time],  # A single record at the evaluation time.
                solver["scheme"],  # Time stepping scheme of Section 4.1.
                float(solver["positivity_tolerance"]),  # Relative tolerance of the monitor.
            )  # Forward solution at the current resolution.
            restricted = restrict_to_grid_2d(reference_state, int(n_points))  # Reference on grid.
            # Relative error of this run against the restricted reference solution.
            error = float(np.linalg.norm(result["density"][0] - restricted) / np.linalg.norm(restricted))
            order = None  # Observed order of convergence between two successive grids.
            if previous_error is not None and error > 0.0:  # An order needs two successive errors.
                order = float(np.log2(previous_error / error))  # Order implied by a halved width.
            records.append(  # One row of the two-dimensional convergence table.
                {  # Diagnostics of the forward solve at the current resolution.
                    "kernel": kernel,  # Detection kernel family of the run.
                    "n_points": int(n_points),  # Grid size in each coordinate direction.
                    "time_step": float(settings["time_step"]),  # Time step of the run.
                    "relative_error": error,  # Relative error against the reference solution.
                    "observed_order": "" if order is None else order,  # Observed order.
                    "mass_error": result["mass_error"],  # Relative mass conservation error.
                    "min_density": result["min_density"],  # Smallest density observed.
                    "wall_time": result["wall_time"],  # Wall clock cost of the forward solve.
                    "steps": result["steps"],  # Number of time steps performed.
                }
            )  # Row appended to the convergence table.
            previous_error = error  # Remember the error for the next refinement level.
    summary = Path("results/summary") / f"{config['experiment']}_convergence_2d.csv"  # Summary.
    write_csv(summary, records, list(records[0].keys()))  # Write the tracked summary table.
    print(f"wrote {summary} with {len(records)} row(s)")  # Report the location of the summary.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the experiment and propagate the exit status.
