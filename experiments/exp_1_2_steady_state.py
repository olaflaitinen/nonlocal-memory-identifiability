# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Agreement between the long time transient solution, the fixed-point
# steady state solver and the positivity-preserving finite-volume scheme. The
# script also records the transient snapshots that make up Fig. 2c.
# Manuscript: Section 5.1, Experiment 1.2; Fig. 2c.
# Inputs: an experiment configuration. Outputs: an HDF5 archive of the
# snapshots and a tracked summary under results/summary/.

import argparse  # Command line interface of the experiment script.
import sys  # Process exit status of the experiment script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and elementary functions.

from nmi.config import load_experiment_config  # Configuration loader with base inheritance.
from nmi.design import initial_density, table3_values  # Initial data and Table 3 parameters.
from nmi.finite_volume import simulate_fv_1d  # Positivity-preserving finite-volume scheme.
from nmi.io import ensure_dir, run_metadata, write_csv, write_hdf5  # Output writers.
from nmi.spectral_1d import simulate_1d  # Pseudo-spectral forward solver.
from nmi.steady_state import fixed_point_steady_state, identity_residual  # Steady state solver.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Steady state agreement")  # Argument parser.
    parser.add_argument("--config", required=True, help="experiment configuration")  # Configuration.
    parser.add_argument("--output-dir", default=None, help="raw output directory")  # Destination.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# Entry point of the steady state agreement study.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    config = load_experiment_config(arguments.config)  # Merged and validated configuration.
    settings = config["steady_state"]  # Settings of the steady state agreement study.
    design = config["design"]  # Block that describes the synthetic design.
    solver = config["solver"]  # Block that describes the numerical resolutions.
    # Directory that receives the archives of the snapshots of Fig. 2c.
    directory = ensure_dir(arguments.output_dir or Path("results/raw") / str(config["experiment"]))
    n_points = int(settings["n_points"])  # Grid size of the transient and fixed-point solves.
    length = float(design["domain_length"])  # Length L of the periodic domain.
    records = []  # Accumulator for the rows of the agreement table.
    for kernel in design["kernels"]:  # Repeat the study for each detection kernel family.
        for radius in design["perceptual_ranges"]:  # Repeat the study for each perceptual range.
            values = table3_values(kernel, float(radius), design)  # Parameter values of Table 3.
            params = {  # Model parameters of equation (1) for this condition.
                "diffusion": float(design["diffusion"]),  # Diffusion rate d of the condition.
                "alpha": float(values["alpha"]),  # Advection strength alpha of the condition.
                "beta": float(design["beta"]),  # Memory uptake rate beta of the condition.
                "memory_decay": float(design["memory_decay"]),  # Memory decay rate mu.
                "radius": float(radius),  # Perceptual range R of the condition.
            }
            transient = simulate_1d(  # Long time transient solve of model (1).
                params,  # Model parameters of the condition.
                kernel,  # Detection kernel family of the condition.
                initial_density(design, n_points),  # Initial density of Section 4.2.
                float(design["initial_map"]),  # Initial cognitive map of Section 4.2.
                float(settings["t_final"]),  # End of the transient solve.
                float(settings["time_step"]),  # Time step of the transient solve.
                n_points,  # Grid size of the transient solve.
                length,  # Length L of the periodic domain.
                settings["record_times"],  # Times at which snapshots are recorded for Fig. 2c.
                solver["scheme"],  # Time stepping scheme of Section 4.1.
                float(solver["positivity_tolerance"]),  # Relative tolerance of the monitor.
            )  # Transient solution that approaches the steady state.
            ratio = float(values["aggregation_ratio"])  # Aggregation ratio kappa of the condition.
            fixed = fixed_point_steady_state(  # Steady state on the branch selected by the transient.
                ratio,  # Aggregation ratio kappa of the condition.
                float(radius),  # Perceptual range R of the condition.
                kernel,  # Detection kernel family of the condition.
                np.maximum(transient["final_density"], 1.0e-12),  # Transient state as the iterate.
                length,  # Length L of the periodic domain.
                float(solver.get("fixed_point_damping", 0.5)),  # Damping of the solver.
                float(solver.get("fixed_point_tolerance", 1.0e-12)),  # Tolerance of the solver.
                int(solver.get("fixed_point_max_iter", 20000)),  # Iteration budget of the solver.
            )  # Converged steady state of equation (7).
            # Largest difference between the transient state and the fixed point.
            difference = float(np.max(np.abs(fixed["density"] - transient["final_density"])))
            scale = float(np.max(np.abs(fixed["density"])))  # Scale used for the relative measure.
            volume = simulate_fv_1d(  # Independent positivity-preserving comparison solve.
                params,  # Model parameters of the condition.
                kernel,  # Detection kernel family of the condition.
                initial_density(design, n_points),  # Initial density of Section 4.2.
                float(design["initial_map"]),  # Initial cognitive map of Section 4.2.
                float(settings["fv_t_final"]),  # End of the finite-volume comparison solve.
                float(settings["fv_time_step"]),  # Time step of the finite-volume solve.
                length,  # Length L of the periodic domain.
                "periodic",  # Periodic boundary condition of the comparison solve.
                [float(settings["fv_t_final"])],  # A single record at the end of the solve.
            )  # Finite-volume solution used to validate the pseudo-spectral scheme.
            spectral_short = simulate_1d(  # Pseudo-spectral solve over the same short window.
                params,  # Model parameters of the condition.
                kernel,  # Detection kernel family of the condition.
                initial_density(design, n_points),  # Initial density of Section 4.2.
                float(design["initial_map"]),  # Initial cognitive map of Section 4.2.
                float(settings["fv_t_final"]),  # End of the comparison window.
                float(settings["time_step"]),  # Time step of the pseudo-spectral solve.
                n_points,  # Grid size of the pseudo-spectral solve.
                length,  # Length L of the periodic domain.
                [float(settings["fv_t_final"])],  # A single record at the end of the window.
                solver["scheme"],  # Time stepping scheme of Section 4.1.
                float(solver["positivity_tolerance"]),  # Relative tolerance of the monitor.
            )  # Pseudo-spectral solution over the finite-volume comparison window.
            # Largest difference between the two independent schemes over the window.
            scheme_difference = float(np.max(np.abs(volume["final_density"] - spectral_short["final_density"])))
            label = f"k-{kernel}_R-{float(radius):.3f}"  # Stable label of this condition.
            write_hdf5(  # Archive of the snapshots that make up Fig. 2c.
                directory / f"steady_{label}.h5",  # Archive of this condition.
                {  # Arrays stored in the archive of this condition.
                    "record_times": np.asarray(transient["times"]),  # Times of the snapshots.
                    "snapshots": transient["density"],  # Transient snapshots of the solve.
                    "steady_state": fixed["density"],  # Steady state of the fixed-point solver.
                    "finite_volume": volume["final_density"],  # Finite-volume comparison state.
                },
                run_metadata(config, {"condition": label}),  # Metadata of this archive.
            )  # Archive written for the figure script of Fig. 2.
            records.append(  # One row of the agreement table of Experiment 1.2.
                {  # Summary of the agreement between the three solvers.
                    "kernel": kernel,  # Detection kernel family of the condition.
                    "radius": float(radius),  # Perceptual range R of the condition.
                    "aggregation_ratio": ratio,  # Aggregation ratio kappa of the condition.
                    # Residual of the stationary identity of equation (7) at the steady state.
                    "identity_residual": identity_residual(fixed["density"], ratio, float(radius), kernel, length),
                    "transient_vs_fixed_point": difference / scale,  # Relative difference.
                    "spectral_vs_finite_volume": scheme_difference,  # Difference of the schemes.
                    "dominant_mode": fixed["dominant_mode"],  # Dominant mode of the steady state.
                    "fixed_point_iterations": fixed["iterations"],  # Iterations of the solver.
                    "min_density_transient": transient["min_density"],  # Smallest transient density.
                    "min_density_finite_volume": volume["min_density"],  # Smallest volume density.
                    "mass_error_transient": transient["mass_error"],  # Mass error of the transient.
                    "mass_error_finite_volume": volume["mass_error"],  # Mass error of the volume.
                }
            )  # Row appended to the agreement table.
            print(f"steady state {label}: relative difference {difference / scale:.3e}")  # Log.
    summary = Path("results/summary") / f"{config['experiment']}_steady_state.csv"  # Summary path.
    write_csv(summary, records, list(records[0].keys()))  # Write the tracked summary table.
    print(f"wrote {summary} with {len(records)} row(s)")  # Report the location of the summary.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the experiment and propagate the exit status.
