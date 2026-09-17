# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Measure the cost of one forward solve at the candidate resolutions,
# before any experiment is run, so that the compute budget of the sampling
# experiments can be planned.
# Manuscript: Section 4.1, performance; Section 4.6.
# Inputs: optional command line arguments that set the resolutions.
# Outputs: a report on standard output.

import argparse  # Command line interface of the pre-experiment check.
import sys  # Process exit status of the pre-experiment check.

import numpy as np  # Numerical arrays and summary statistics.

from nmi.grids import periodic_grid_1d  # Equispaced grid of the periodic domain.
from nmi.spectral_1d import simulate_1d  # Pseudo-spectral forward solver.

# Length of the periodic domain used by the timing.
DOMAIN_LENGTH = 1.0  # Nondimensionalised domain length of the synthetic experiments.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Forward solve timing")  # Argument parser.
    parser.add_argument("--grids", default="128,256,512,1024")  # Grid sizes that are timed.
    parser.add_argument("--time-step", type=float, default=0.005)  # Time step of the solves.
    parser.add_argument("--t-final", type=float, default=100.0)  # End of the timed solves.
    parser.add_argument("--repeats", type=int, default=3)  # Repetitions of each timing.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# Entry point of the timing check.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    grids = [int(value) for value in str(arguments.grids).split(",")]  # Grid sizes to be timed.
    params = {  # Model parameters of the timed forward solves.
        "diffusion": 0.01,  # Diffusion rate d of the timed solves.
        "alpha": 0.01398,  # Advection strength alpha, taken from Table 3 at R = 0.15.
        "beta": 1.0,  # Memory uptake rate beta, set to one without loss of generality.
        "memory_decay": 1.0,  # Memory decay rate mu of the timed solves.
        "radius": 0.15,  # Perceptual range R of the timed solves.
    }
    print("grid   steps    median (s)   seconds per step")  # Header of the timing report.
    for n_points in grids:  # Time one forward solve at each candidate grid size.
        grid = periodic_grid_1d(n_points, DOMAIN_LENGTH)  # Grid points of the periodic domain.
        initial = 1.0 + 0.05 * np.cos(2.0 * np.pi * grid / DOMAIN_LENGTH)  # Initial density.
        timings = []  # Wall clock cost of each repetition of the timed solve.
        steps = 0  # Number of time steps performed by the timed solve.
        for _ in range(int(arguments.repeats)):  # Repeat the timing so that a median is available.
            result = simulate_1d(  # Timed pseudo-spectral forward solve.
                params,  # Model parameters of the timed solve.
                "tophat",  # Detection kernel family of the timed solve.
                initial,  # Initial density of the timed solve.
                0.0,  # Initial cognitive map of the timed solve.
                float(arguments.t_final),  # End of the timed solve.
                float(arguments.time_step),  # Time step of the timed solve.
                int(n_points),  # Grid size of the timed solve.
                DOMAIN_LENGTH,  # Length L of the periodic domain.
            )  # Result of one timed forward solve.
            timings.append(result["wall_time"])  # Record the cost of this repetition.
            steps = result["steps"]  # Number of time steps performed by the timed solve.
        median = float(np.median(timings))  # Median cost over the repetitions of this grid size.
        print(f"{n_points:5d} {steps:7d} {median:12.3f} {median / max(1, steps):18.3e}")  # Report.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the timing and propagate the exit status.
