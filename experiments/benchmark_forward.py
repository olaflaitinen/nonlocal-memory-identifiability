# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Measure the cost of one forward solve at the configured resolutions,
# with and without the optional Numba acceleration of the finite-volume solver,
# so that the compute budget of the sampling experiments can be planned.
# Manuscript: Section 4.1, performance; Section 4.6.
# Inputs: an experiment configuration. Outputs: a tracked summary under
# results/summary/.

import argparse  # Command line interface of the experiment script.
import sys  # Process exit status of the experiment script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and summary statistics.

from nmi.config import load_experiment_config  # Configuration loader with base inheritance.
from nmi.design import initial_density, table3_values  # Initial data and Table 3 parameters.
from nmi.finite_volume import simulate_fv_1d  # Positivity-preserving finite-volume scheme.
from nmi.io import write_csv  # Writer of the tracked summary table.
from nmi.spectral_1d import simulate_1d  # Pseudo-spectral forward solver.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Forward solve benchmark")  # Argument parser.
    parser.add_argument("--config", required=True, help="experiment configuration")  # Configuration.
    parser.add_argument("--output-dir", default=None, help="raw output directory")  # Destination.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# Entry point of the forward solve benchmark.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    config = load_experiment_config(arguments.config)  # Merged and validated configuration.
    settings = config.get("benchmark", {})  # Settings of the benchmark, with defaults below.
    design = config["design"]  # Block that describes the synthetic design.
    solver = config["solver"]  # Block that describes the numerical resolutions.
    grids = settings.get("grids", [128, 256, 512])  # Grid sizes at which the solve is timed.
    repeats = int(settings.get("repeats", 3))  # Number of repetitions of each timing.
    t_final = float(settings.get("t_final", design["observation_time"]))  # End of the solve.
    radius = 0.15  # Perceptual range used by the benchmark, a mid range value of Table 3.
    values = table3_values("tophat", radius, design)  # Parameter values of Table 3.
    params = {  # Model parameters of equation (1) used by the benchmark.
        "diffusion": float(design["diffusion"]),  # Diffusion rate d of the benchmark.
        "alpha": float(values["alpha"]),  # Advection strength alpha of the benchmark.
        "beta": float(design["beta"]),  # Memory uptake rate beta of the benchmark.
        "memory_decay": float(design["memory_decay"]),  # Memory decay rate mu.
        "radius": radius,  # Perceptual range R of the benchmark.
    }
    time_step = float(solver["inference"]["time_step"])  # Time step of the timed solves.
    records = []  # Accumulator for the rows of the benchmark table.
    for n_points in grids:  # Time one forward solve at each configured grid size.
        timings = []  # Wall clock cost of each repetition of the pseudo-spectral solve.
        for _ in range(repeats):  # Repeat the timing so that a median can be reported.
            result = simulate_1d(  # Pseudo-spectral forward solve at the current grid size.
                params,  # Model parameters of the benchmark.
                "tophat",  # Detection kernel family of the benchmark.
                initial_density(design, int(n_points)),  # Initial density of Section 4.2.
                float(design["initial_map"]),  # Initial cognitive map of Section 4.2.
                t_final,  # End of the timed forward solve.
                time_step,  # Time step of the timed forward solve.
                int(n_points),  # Grid size of the timed forward solve.
                float(design["domain_length"]),  # Length L of the periodic domain.
                [t_final],  # A single record at the end of the window.
                solver["scheme"],  # Time stepping scheme of Section 4.1.
                float(solver["positivity_tolerance"]),  # Relative tolerance of the monitor.
            )  # Timed pseudo-spectral forward solve.
            timings.append(result["wall_time"])  # Record the cost of this repetition.
        for use_numba in (False, True):  # Time the finite-volume solver in both modes.
            volume = simulate_fv_1d(  # Finite-volume solve over a short window.
                params,  # Model parameters of the benchmark.
                "tophat",  # Detection kernel family of the benchmark.
                initial_density(design, int(n_points)),  # Initial density of Section 4.2.
                float(design["initial_map"]),  # Initial cognitive map of Section 4.2.
                min(t_final, 1.0),  # Short window, since the explicit scheme is expensive.
                # Explicit time step chosen well inside the diffusive stability limit.
                0.2 * (float(design["domain_length"]) / float(n_points)) ** 2 / design["diffusion"],
                float(design["domain_length"]),  # Length L of the periodic domain.
                "periodic",  # Periodic boundary condition of the benchmark.
                [min(t_final, 1.0)],  # A single record at the end of the short window.
                use_numba,  # Whether the compiled flux update is used.
            )  # Timed finite-volume forward solve.
            records.append(  # One row of the benchmark table for the finite-volume solver.
                {  # Cost of one finite-volume solve at the current grid size.
                    "solver": "finite_volume",  # Name of the timed solver.
                    "n_points": int(n_points),  # Grid size of the timed solve.
                    "use_numba": use_numba,  # Whether the compiled flux update was used.
                    "median_wall_time": volume["wall_time"],  # Wall clock cost of the solve.
                    "steps": volume["steps"],  # Number of time steps performed.
                    "seconds_per_step": volume["wall_time"] / max(1, volume["steps"]),  # Cost.
                }
            )  # Row appended to the benchmark table.
        records.append(  # One row of the benchmark table for the pseudo-spectral solver.
            {  # Cost of one pseudo-spectral solve at the current grid size.
                "solver": "pseudo_spectral",  # Name of the timed solver.
                "n_points": int(n_points),  # Grid size of the timed solve.
                "use_numba": False,  # The pseudo-spectral solver is not accelerated by Numba.
                "median_wall_time": float(np.median(timings)),  # Median cost over the repetitions.
                "steps": result["steps"],  # Number of time steps performed.
                "seconds_per_step": float(np.median(timings)) / max(1, result["steps"]),  # Cost.
            }
        )  # Row appended to the benchmark table.
        # Report the median cost of one pseudo-spectral solve at this grid size.
        print(f"grid {int(n_points)}: median {float(np.median(timings)):.3f} s per spectral solve")
    summary = Path("results/summary") / f"{config['experiment']}_benchmark_forward.csv"  # Summary.
    write_csv(summary, records, list(records[0].keys()))  # Write the tracked summary table.
    print(f"wrote {summary} with {len(records)} row(s)")  # Report the location of the summary.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the experiment and propagate the exit status.
