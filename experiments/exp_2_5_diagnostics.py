# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Collect the convergence diagnostics of every one-dimensional
# condition, apply the criteria of Section 4.4 and write the list of conditions
# that must be rerun with a longer chain.
# Manuscript: Section 5.3, Experiment 2.5.
# Inputs: an experiment configuration and the chains of Experiment 2.4.
# Outputs: a tracked diagnostics table and a rerun list under results/summary/.

import argparse  # Command line interface of the experiment script.
import sys  # Process exit status of the experiment script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and elementary functions.

from nmi.config import load_experiment_config  # Configuration loader with base inheritance.
from nmi.design import design_1d  # Construction of the factorial design.
from nmi.diagnostics import convergence_statistics, has_converged, rerun_list  # Diagnostics.
from nmi.io import read_hdf5, write_csv  # Reader of the chain archives and summary writer.

# Names of the sampled parameters, in the order used by every sampler.
PARAMETER_NAMES = ("diffusion", "advection", "memory_decay", "radius")  # Reduced parameter vector.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Convergence diagnostics")  # Argument parser.
    parser.add_argument("--config", required=True, help="experiment configuration")  # Configuration.
    parser.add_argument("--output-dir", default=None, help="raw output directory")  # Destination.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# Entry point of the diagnostics experiment.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    config = load_experiment_config(arguments.config)  # Merged and validated configuration.
    design = design_1d(config)  # Factorial design of the one-dimensional experiments.
    directory = Path(arguments.output_dir or Path("results/raw") / str(config["experiment"]))
    burn_in = float(config["mcmc"]["burn_in_fraction"])  # Fraction discarded as burn-in.
    records = []  # Accumulator for the rows of the diagnostics table.
    for condition in design["transient"]:  # Inspect the chains of each condition in turn.
        archive = directory / f"chains_{condition['index']:03d}.h5"  # Archive of this condition.
        if not archive.is_file():  # The chains of this condition have not been produced yet.
            print(f"missing chains for condition {condition['index']:03d}")  # Report the gap.
            continue  # Continue with the next condition of the design.
        arrays, _ = read_hdf5(archive)  # Chains of this condition and their metadata.
        stacked = arrays["chains"]  # Chains of shape (chains, iterations, parameters).
        kept = np.exp(stacked[:, int(burn_in * stacked.shape[1]) :, :])  # Retained draws.
        statistics = convergence_statistics(kept, PARAMETER_NAMES)  # Convergence diagnostics.
        record = {  # One row of the diagnostics table for this condition.
            "condition_index": condition["index"],  # Stable integer index of the condition.
            "identifier": condition["identifier"],  # Stable textual label of the condition.
            "n_chains": int(stacked.shape[0]),  # Number of chains of this condition.
            "n_iterations": int(stacked.shape[1]),  # Number of iterations of each chain.
            "converged": has_converged(statistics),  # Whether the criteria of Section 4.4 hold.
        }
        for name in PARAMETER_NAMES:  # Record both diagnostics of every sampled parameter.
            record[f"rhat_{name}"] = statistics[name]["rhat"]  # Rank-normalised split R-hat.
            record[f"ess_{name}"] = statistics[name]["ess_bulk"]  # Bulk effective sample size.
        record["max_rhat"] = max(statistics[name]["rhat"] for name in PARAMETER_NAMES)  # Worst.
        record["min_ess"] = min(statistics[name]["ess_bulk"] for name in PARAMETER_NAMES)  # Worst.
        records.append(record)  # Row appended to the diagnostics table.
    if not records:  # No chains were available, so nothing can be diagnosed.
        print("no chains were found, run exp_2_4_mcmc_1d.py first")  # Report the missing input.
        return 0  # Signal success, since the absence of chains is not an error here.
    summary = Path("results/summary") / f"{config['experiment']}_diagnostics.csv"  # Summary path.
    write_csv(summary, records, list(records[0].keys()))  # Write the tracked diagnostics table.
    failures = rerun_list(records)  # Identifiers of the conditions that failed the criteria.
    rerun = Path("results/summary") / f"{config['experiment']}_rerun_list.csv"  # Rerun list path.
    rows = [{"identifier": name} for name in failures]  # One row per condition to be rerun.
    write_csv(rerun, rows or [{"identifier": ""}], ["identifier"])  # Write the rerun list.
    print(f"wrote {summary} with {len(records)} row(s) and {len(failures)} condition(s) to rerun")
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the experiment and propagate the exit status.
