# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Convergence study of the one-dimensional pseudo-spectral solver. The
# relative error against a fine reference solution is measured as the grid is
# refined at a fixed time step and as the time step is refined at a fixed grid,
# together with the mass conservation error and the cost of a forward solve.
# Manuscript: Section 5.1, Experiment 1.1; Table 5 and Fig. 2a,b.
# Inputs: an experiment configuration. Outputs: a tracked summary under
# results/summary/ that feeds Table 5 and Fig. 2.

import argparse  # Command line interface of the experiment script.
import sys  # Process exit status of the experiment script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and elementary functions.

from nmi.config import load_experiment_config  # Configuration loader with base inheritance.
from nmi.design import initial_density, table3_values  # Initial data and Table 3 parameters.
from nmi.grids import periodic_grid_1d  # Equispaced grid of the periodic domain.
from nmi.io import write_csv  # Writer of the tracked summary table.
from nmi.spectral_1d import simulate_1d  # Pseudo-spectral forward solver.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="One-dimensional convergence")  # Argument parser.
    parser.add_argument("--config", required=True, help="experiment configuration")  # Configuration.
    parser.add_argument("--output-dir", default=None, help="raw output directory")  # Destination.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# Interpolate a solution from its own grid onto a coarser grid by subsampling.
# Arguments:
#   values (numpy.ndarray): the solution on the fine grid.
#   n_coarse (int): number of points of the coarse grid.
# Returns:
#   numpy.ndarray: the solution restricted to the coarse grid points.
def restrict_to_grid(values, n_coarse):
    n_fine = values.shape[-1]  # Number of points of the fine grid.
    if n_fine % n_coarse != 0:  # The coarse grid must be a subgrid of the fine grid.
        raise ValueError(f"The grid of {n_fine} points is not a multiple of {n_coarse}")  # Reject.
    return values[..., :: n_fine // n_coarse]  # Subsample the fine solution onto the coarse grid.


# Run one forward solve and return the recorded state at the evaluation time.
# Arguments:
#   params (dict): model parameters of equation (1).
#   kernel (str): detection kernel family.
#   config (dict): the configuration mapping of the run.
#   n_points (int): number of grid points of the solve.
#   time_step (float): time step of the solve.
#   evaluation_time (float): time at which the state is recorded.
# Returns:
#   dict: the recorded state together with the diagnostics of the solve.
def run_solve(params, kernel, config, n_points, time_step, evaluation_time):
    design = config["design"]  # Block that describes the synthetic design.
    solver = config["solver"]  # Block that describes the numerical resolutions.
    result = simulate_1d(  # Forward solve at the requested grid and time step.
        params,  # Model parameters of equation (1).
        kernel,  # Detection kernel family of the solve.
        initial_density(design, n_points),  # Initial density of Section 4.2.
        float(design["initial_map"]),  # Initial cognitive map of Section 4.2.
        float(evaluation_time),  # End of the forward solve of this run.
        float(time_step),  # Time step of this run.
        int(n_points),  # Grid size of this run.
        float(design["domain_length"]),  # Length L of the periodic domain.
        [float(evaluation_time)],  # A single record at the evaluation time.
        solver["scheme"],  # Time stepping scheme of Section 4.1.
        float(solver["positivity_tolerance"]),  # Relative tolerance of the positivity monitor.
    )  # Forward solution at the requested resolution.
    return result  # Recorded state together with the diagnostics of the solve.


# Entry point of the one-dimensional convergence study.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    config = load_experiment_config(arguments.config)  # Merged and validated configuration.
    settings = config["convergence"]  # Settings of the one-dimensional convergence study.
    design = config["design"]  # Block that describes the synthetic design.
    radius = float(settings["radius"])  # Perceptual range used by the convergence study.
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
        reference = run_solve(  # Fine reference solve used as the ground truth.
            params,  # Model parameters of the condition.
            kernel,  # Detection kernel family of the condition.
            config,  # Configuration mapping of the run.
            int(settings["reference_grid"]),  # Grid size of the reference solution.
            float(settings["reference_time_step"]),  # Time step of the reference solution.
            evaluation_time,  # Time at which the reference state is recorded.
        )  # Reference solution against which every error is measured.
        reference_state = reference["density"][0]  # Reference state at the evaluation time.
        previous_error = None  # Error of the previous grid, used for the observed order.
        for n_points in settings["grids"]:  # Refine the grid at the fixed reference time step.
            # Forward solve on the current grid at the reference time step.
            result = run_solve(params, kernel, config, int(n_points), float(settings["reference_time_step"]), evaluation_time)
            restricted = restrict_to_grid(reference_state, int(n_points))  # Reference on the grid.
            # Relative error of this run against the restricted reference solution.
            error = float(np.linalg.norm(result["density"][0] - restricted) / np.linalg.norm(restricted))
            order = None  # Observed order of convergence between two successive grids.
            if previous_error is not None and error > 0.0:  # An order can only be formed from two errors.
                order = float(np.log2(previous_error / error))  # Order implied by a halved cell width.
            step = float(settings["reference_time_step"])  # Time step of the grid refinement study.
            records.append(_record(kernel, "grid", int(n_points), step, error, order, result))  # One row.
            previous_error = error  # Remember the error for the next refinement level.
        previous_error = None  # Reset the error before the time step refinement study.
        for time_step in settings["time_steps"]:  # Refine the time step at the inference grid.
            n_points = int(config["solver"]["inference"]["n_points"])  # Grid of the refinement study.
            # Forward solve on the inference grid at the current time step.
            result = run_solve(params, kernel, config, n_points, float(time_step), evaluation_time)
            restricted = restrict_to_grid(reference_state, n_points)  # Reference on the same grid.
            # Relative error of this run against the restricted reference solution.
            error = float(np.linalg.norm(result["density"][0] - restricted) / np.linalg.norm(restricted))
            order = None  # Observed order of convergence between two successive time steps.
            if previous_error is not None and error > 0.0:  # An order needs two successive errors.
                order = float(np.log2(previous_error / error))  # Order implied by a halved time step.
            # One row of the time step refinement study.
            records.append(_record(kernel, "time_step", n_points, float(time_step), error, order, result))
            previous_error = error  # Remember the error for the next refinement level.
    summary = Path("results/summary") / f"{config['experiment']}_convergence_1d.csv"  # Summary.
    write_csv(summary, records, list(records[0].keys()))  # Write the tracked summary table.
    print(f"wrote {summary} with {len(records)} row(s)")  # Report the location of the summary.
    return 0  # Signal success to the caller.


# Build one row of the convergence table.
# Arguments:
#   kernel (str): detection kernel family of the run.
#   refinement (str): "grid" or "time_step", naming the refined quantity.
#   n_points (int): grid size of the run.
#   time_step (float): time step of the run.
#   error (float): relative error against the reference solution.
#   order (float): observed order of convergence, or None.
#   result (dict): the diagnostics of the forward solve.
# Returns:
#   dict: one row of the convergence table.
def _record(kernel, refinement, n_points, time_step, error, order, result):
    return {  # One row of the convergence table of Table 5.
        "kernel": kernel,  # Detection kernel family of the run.
        "refinement": refinement,  # Quantity that was refined in this part of the study.
        "n_points": n_points,  # Grid size of the run.
        "time_step": time_step,  # Time step of the run.
        "relative_error": error,  # Relative error against the reference solution.
        "observed_order": "" if order is None else order,  # Observed order of convergence.
        "mass_error": result["mass_error"],  # Relative mass conservation error of the run.
        "min_density": result["min_density"],  # Smallest density observed during the run.
        "wall_time": result["wall_time"],  # Wall clock cost of the forward solve.
    }


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the experiment and propagate the exit status.
