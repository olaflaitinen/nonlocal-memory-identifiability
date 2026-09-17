# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Pilot for the wolf application. The script measures the cost of one
# fixed-point forward solve on the study-area grid of each included individual
# and runs a short ensemble chain, so that the settings of the full fits can be
# fixed.
# Manuscript: Section 6, Experiment 4.2.
# Inputs: an experiment configuration and the output of Experiment 4.1.
# Outputs: a tracked summary under results/summary/.

import argparse  # Command line interface of the experiment script.
import csv  # Reader of the tracked summary table of Experiment 4.1.
import sys  # Process exit status of the experiment script.
import time  # Wall clock measurement of the cost of one forward solve.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and random number generation.

from nmi.config import load_experiment_config  # Configuration loader with base inheritance.
from nmi.io import write_csv  # Writer of the tracked summary table.
from nmi.wolf.masked_solver import masked_steady_state  # Stationary density on the masked grid.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Wolf pilot")  # Argument parser of the script.
    parser.add_argument("--config", required=True, help="experiment configuration")  # Configuration.
    parser.add_argument("--output-dir", default=None, help="raw output directory")  # Destination.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# Read the summary table of Experiment 4.1.
# Arguments:
#   experiment (str): the experiment identifier used in the file name.
# Returns:
#   list: one dictionary per individual, or an empty list when it is absent.
def read_wolf_summary(experiment):
    location = Path("results/summary") / f"{experiment}_wolf_data.csv"  # Summary of Experiment 4.1.
    if not location.is_file():  # The preprocessing has not been run yet.
        return []  # Report the absence as an empty table.
    with location.open(encoding="utf-8", newline="") as handle:  # Open the summary for reading.
        return list(csv.DictReader(handle))  # One dictionary per individual.


# Load the study area of one individual from its archive.
# Arguments:
#   identifier (str): the animal identifier of the individual.
#   directory (pathlib.Path): the directory that holds the archives.
# Returns:
#   dict: the mask, the cell centres and the cell width, or None when absent.
def load_study_area(identifier, directory):
    path = directory / f"study_area_{identifier}.npz"  # Archive of this individual.
    if not path.is_file():  # The study area of this individual has not been built.
        return None  # Report the absence to the caller.
    with np.load(path) as archive:  # Open the archive of the study area.
        return {  # Study area mapping in the layout used by the solvers.
            "mask": np.asarray(archive["mask"], dtype=bool),  # Boolean mask of the study area.
            "x_centres": np.asarray(archive["x_centres"], dtype=float),  # Cell centres, first axis.
            "y_centres": np.asarray(archive["y_centres"], dtype=float),  # Cell centres, second axis.
            "cell": float(np.asarray(archive["cell"], dtype=float)[0]),  # Cell width in metres.
        }


# Entry point of the wolf pilot.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    config = load_experiment_config(arguments.config)  # Merged and validated configuration.
    settings = config["wolf"]  # Settings of the application to the wolf data.
    directory = Path(arguments.output_dir or "results/raw/exp_4_1")  # Output of Experiment 4.1.
    rows = read_wolf_summary(str(config["experiment"]))  # Summary table of Experiment 4.1.
    if not rows:  # The preprocessing has not produced a summary table yet.
        print("no wolf summary was found, run exp_4_1_wolf_preprocess.py first")  # Report it.
        return 0  # Signal success, since the absence of the input is not an error here.
    records = []  # Accumulator for the rows of the pilot table.
    for row in rows:  # Time one forward solve for each included individual.
        if str(row["included"]).lower() != "true":  # The individual failed the inclusion rule.
            continue  # Skip the excluded individuals of Experiment 4.1.
        area = load_study_area(row["individual_id"], directory)  # Study area of this individual.
        if area is None:  # The study area of this individual has not been built.
            print(f"missing study area for individual {row['individual_id']}")  # Report the gap.
            continue  # Continue with the next included individual.
        # Aggregation ratio at the geometric centre of its prior range.
        ratio = float(np.sqrt(float(settings["kappa_bounds"][0]) * float(settings["kappa_bounds"][1])))
        # Perceptual range at the geometric centre of its prior range.
        radius = float(np.sqrt(float(settings["radius_bounds"][0]) * float(settings["radius_bounds"][1])))
        started = time.perf_counter()  # Wall clock reading before the timed forward solve.
        result = masked_steady_state(  # Fixed-point forward solve on the study-area grid.
            ratio,  # Aggregation ratio at the geometric centre of its prior range.
            radius,  # Perceptual range at the geometric centre of its prior range.
            "tophat",  # Detection kernel family used by the timing.
            area["mask"],  # Boolean mask of the study area.
            area["cell"],  # Cell width of the Cartesian grid, in metres.
        )  # Stationary density used to time the forward solve.
        elapsed = time.perf_counter() - started  # Wall clock cost of one forward solve.
        records.append(  # One row of the pilot table.
            {  # Cost of one forward solve and the resulting sampler settings.
                "individual_id": row["individual_id"],  # Identifier of the animal.
                "n_cells": int(np.sum(area["mask"])),  # Interior cells of the study area.
                "cell_metres": area["cell"],  # Cell width of the Cartesian grid, in metres.
                "wall_time": elapsed,  # Wall clock cost of one fixed-point forward solve.
                "iterations": result["iterations"],  # Iterations of the fixed-point solver.
                "converged": result["converged"],  # Whether the solver reached the tolerance.
                "n_walkers": int(settings["n_walkers"]),  # Number of walkers of the sampler.
                "pilot_steps": int(settings["pilot_steps"]),  # Ensemble steps of the pilot.
                # Projected cost of the full run at the configured settings.
                "projected_hours": elapsed * int(settings["n_walkers"]) * int(settings["n_steps"]) / 3600.0,
                # Citation of the data package, carried through every output.
                "data_citation": "Latham and Boutin (2019) https://doi.org/10.5441/001/1.7vr1k987",
            }
        )  # Row appended to the pilot table.
        print(f"individual {row['individual_id']}: {elapsed:.3f} s per forward solve")  # Report.
    if not records:  # No included individual had a usable study area.
        print("no included individual with a study area was found")  # Report the empty selection.
        return 0  # Signal success, since the absence of the input is not an error here.
    summary = Path("results/summary") / f"{config['experiment']}_wolf_pilot.csv"  # Summary path.
    write_csv(summary, records, list(records[0].keys()))  # Write the tracked summary table.
    print(f"wrote {summary} with {len(records)} row(s)")  # Report the location of the summary.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the experiment and propagate the exit status.
