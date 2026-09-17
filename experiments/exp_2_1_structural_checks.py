# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Numerical check of the structural results. The script forms the
# finite-difference sensitivity matrix with respect to the logarithms of the
# full parameter vector and of the reduced vector, reports the singular values
# and the condition number of the Fisher information, and evaluates the
# log-likelihood surfaces over the advection pair and over the diffusion and
# memory decay pair that make up Fig. 3.
# Manuscript: Section 5.2, Experiment 2.1; Proposition 1, Theorem 1(i), Remark 1.
# Inputs: an experiment configuration and the generated synthetic data.
# Outputs: an HDF5 archive of the surfaces and a tracked summary.

import argparse  # Command line interface of the experiment script.
import sys  # Process exit status of the experiment script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and elementary functions.

from nmi.config import load_experiment_config  # Configuration loader with base inheritance.
from nmi.design import design_1d  # Construction of the factorial design.
from nmi.io import ensure_dir, read_condition_data, run_metadata, write_csv, write_hdf5  # Output helpers.
from nmi.likelihood import (  # Forward maps and the Gaussian log-likelihood.
    forward_stationary_loglik,  # Log-likelihood of the stationary observations.
    forward_transient_loglik,  # Log-likelihood of the transient observations.
    predict_transient,  # Prediction of the transient forward map.
    stationary_context,  # Context mapping of the stationary forward map.
    transient_context,  # Context mapping of the transient forward map.
)
from nmi.sensitivity import (  # Local sensitivity analysis of the forward map.
    condition_number,  # Condition number of the Fisher information matrix.
    fisher_information,  # Fisher information of the Gaussian observation model.
    numerical_rank,  # Numerical rank implied by the singular values.
    sensitivity_matrix,  # Finite-difference sensitivity matrix.
    singular_values,  # Singular values of the sensitivity matrix.
)


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Structural checks")  # Argument parser.
    parser.add_argument("--config", required=True, help="experiment configuration")  # Configuration.
    parser.add_argument("--condition-index", type=int, default=None, help="one condition")  # Subset.
    parser.add_argument("--output-dir", default=None, help="raw output directory")  # Destination.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# Evaluate a log-likelihood surface over two parameters on a logarithmic grid.
# Arguments:
#   evaluate (callable): log-likelihood as a function of the log parameters.
#   centre (numpy.ndarray): the true parameters in log space.
#   indices (tuple): positions of the two parameters that vary.
#   half_width (float): half width of the grid in natural logarithm units.
#   n_points (int): number of grid points along each axis.
# Returns:
#   tuple: the two grids and the surface of log-likelihood values.
def log_likelihood_surface(evaluate, centre, indices, half_width, n_points):
    first, second = indices  # Positions of the two parameters that vary on the surface.
    # Logarithmic axis of the first varying parameter, centred on its true value.
    axis_a = np.linspace(centre[first] - half_width, centre[first] + half_width, int(n_points))
    # Logarithmic axis of the second varying parameter, centred on its true value.
    axis_b = np.linspace(centre[second] - half_width, centre[second] + half_width, int(n_points))
    surface = np.full((int(n_points), int(n_points)), -np.inf)  # Storage of the surface values.
    for row, value_a in enumerate(axis_a):  # Vary the first parameter along the rows.
        for column, value_b in enumerate(axis_b):  # Vary the second parameter along the columns.
            point = np.array(centre, dtype=float)  # Start from the true parameter vector.
            point[first] = value_a  # Current value of the first varying parameter.
            point[second] = value_b  # Current value of the second varying parameter.
            surface[row, column] = evaluate(point)  # Log-likelihood at the current point.
    return axis_a, axis_b, surface  # Axes and the evaluated log-likelihood surface.


# Entry point of the structural checks.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    config = load_experiment_config(arguments.config)  # Merged and validated configuration.
    settings = config["structural"]  # Settings of the structural checks.
    design = design_1d(config)  # Factorial design of the one-dimensional experiments.
    index = arguments.condition_index  # Condition index supplied on the command line.
    if index is None:  # No condition was supplied, so the configured one is used.
        index = int(settings["condition_index"]) % len(design["transient"])  # Configured index.
    condition = design["transient"][int(index)]  # Condition inspected by the structural checks.
    data = read_condition_data(config["experiment"], condition["index"])  # Generated data.
    context = transient_context(condition, config)  # Context of the transient forward map.
    truth = np.log(  # True parameters of the condition in log space.
        [  # Reduced parameter vector in the order used by the samplers.
            float(condition["diffusion"]),  # True diffusion rate d.
            float(condition["advection"]),  # True combined advection strength gamma.
            float(condition["memory_decay"]),  # True memory decay rate mu.
            float(condition["radius"]),  # True perceptual range R.
        ]
    )  # Reduced parameter vector of Proposition 1.
    step = float(settings["finite_difference_step"])  # Relative step of the central difference.
    # Sensitivity matrix of the reduced parameter vector of Proposition 1.
    reduced = sensitivity_matrix(lambda point: predict_transient(point, context), truth, step)
    reduced_values = singular_values(reduced)  # Singular values of the reduced sensitivity matrix.
    reduced_information = fisher_information(reduced, data["sigma"])  # Fisher information matrix.
    # True full parameter vector, which separates the advection strength from the uptake rate.
    full_truth = np.concatenate([truth[:1], np.log([float(condition["alpha"]), float(condition["beta"])]), truth[2:]])
    # Sensitivity matrix of the full parameter vector, which is rank deficient.
    full = sensitivity_matrix(lambda point: _predict_full(point, context), full_truth, step)
    full_values = singular_values(full)  # Singular values of the full sensitivity matrix.
    full_information = fisher_information(full, data["sigma"])  # Fisher information matrix.
    # Directory that receives the archive of the surfaces of Fig. 3.
    directory = ensure_dir(arguments.output_dir or Path("results/raw") / str(config["experiment"]))
    transient_data = {  # Data mapping used by the transient log-likelihood.
        "observations": data["observations"],  # Noisy transient observations.
        "sigma": data["sigma"],  # Standard deviation of the observation noise.
    }
    axis_alpha, axis_beta, surface_ab = log_likelihood_surface(  # Surface over the advection pair.
        # Log-likelihood of the transient data as a function of the full parameter vector.
        lambda point: forward_transient_loglik(_reduce_full(point), transient_data, context),
        full_truth,  # True parameters of the condition in log space.
        (1, 2),  # Positions of the advection strength and of the memory uptake rate.
        float(settings["surface_half_width"]),  # Half width of the surface in log units.
        int(settings["surface_points"]),  # Number of grid points along each axis.
    )  # Surface that exhibits the symmetry of Proposition 1.
    stationary_condition = dict(condition)  # Copy used for the stationary forward map.
    stationary_data = {  # Data mapping used by the stationary log-likelihood.
        "observations": data["stationary_observations"],  # Noisy stationary observations.
        "sigma": data["sigma"],  # Standard deviation of the observation noise.
    }
    stationary_ctx = stationary_context(stationary_condition, config)  # Stationary context.
    # True stationary parameters of the condition, in log space.
    stationary_truth = np.log([float(condition["aggregation_ratio"]), float(condition["radius"])])
    axis_d, axis_mu, surface_dmu = _stationary_surface(  # Surface over the diffusion and decay pair.
        stationary_data,  # Data mapping of the stationary observations.
        stationary_ctx,  # Context of the stationary forward map.
        condition,  # Condition whose true parameters centre the surface.
        float(settings["surface_half_width"]),  # Half width of the surface in log units.
        int(settings["surface_points"]),  # Number of grid points along each axis.
    )  # Surface that exhibits the degeneracy of Theorem 1(i).
    write_hdf5(  # Archive of the surfaces that make up Fig. 3.
        directory / f"structural_{condition['index']:03d}.h5",  # Archive of this condition.
        {  # Arrays stored in the archive of this condition.
            "axis_alpha": axis_alpha,  # Logarithm of the advection strength along the first axis.
            "axis_beta": axis_beta,  # Logarithm of the memory uptake rate along the second axis.
            "surface_alpha_beta": surface_ab,  # Log-likelihood surface of Fig. 3a.
            "axis_diffusion": axis_d,  # Logarithm of the diffusion rate along the first axis.
            "axis_memory_decay": axis_mu,  # Logarithm of the memory decay rate along the axis.
            "surface_diffusion_decay": surface_dmu,  # Log-likelihood surface of Fig. 3b.
            "singular_values_reduced": reduced_values,  # Singular values of the reduced matrix.
            "singular_values_full": full_values,  # Singular values of the full matrix.
            "truth_reduced": truth,  # True reduced parameter vector in log space.
            "truth_full": full_truth,  # True full parameter vector in log space.
            "truth_stationary": stationary_truth,  # True stationary parameters in log space.
        },
        run_metadata(config, {"condition": condition["identifier"]}),  # Metadata of this archive.
    )  # Archive written for the figure script of Fig. 3.
    record = {  # Summary row of the structural checks of this condition.
        "condition_index": condition["index"],  # Stable integer index of the condition.
        "identifier": condition["identifier"],  # Stable textual label of the condition.
        "kernel": condition["kernel"],  # Detection kernel family of the condition.
        "radius": condition["radius"],  # Perceptual range R of the condition.
        "noise_level": condition["noise_level"],  # Relative noise level eta of the condition.
        "rank_full": numerical_rank(full_values),  # Numerical rank of the full sensitivity matrix.
        "rank_reduced": numerical_rank(reduced_values),  # Numerical rank of the reduced matrix.
        "condition_number_full": condition_number(full_information),  # Condition number, full.
        "condition_number_reduced": condition_number(reduced_information),  # Reduced condition.
        "smallest_singular_value_full": float(full_values[-1]),  # Smallest singular value, full.
        "smallest_singular_value_reduced": float(reduced_values[-1]),  # Reduced smallest value.
        "largest_singular_value_full": float(full_values[0]),  # Largest singular value, full.
    }
    summary = Path("results/summary") / f"{config['experiment']}_structural.csv"  # Summary path.
    write_csv(summary, [record], list(record.keys()))  # Write the tracked summary table.
    # Report the numerical rank, which is four rather than five by Proposition 1.
    print(f"structural checks: rank {record['rank_full']} of 5 for the full parameter vector")
    print(f"wrote {summary}")  # Report the location of the summary.
    return 0  # Signal success to the caller.


# Prediction of the transient forward map from the full parameter vector.
# Arguments:
#   point (numpy.ndarray): logarithms of (d, alpha, beta, mu, R).
#   context (dict): the context mapping of the transient forward map.
# Returns:
#   numpy.ndarray: the prediction at the sampling points of the design.
def _predict_full(point, context):
    prediction = predict_transient(_reduce_full(point), context)  # Reduced forward map.
    if prediction is None:  # The proposal was rejected by the positivity monitor.
        # A rejected proposal cannot be differentiated, so the failure is reported.
        raise ValueError("The forward solve lost positivity during the sensitivity analysis")
    return prediction  # Prediction at the sampling points of the design.


# Reduce the full parameter vector to the identifiable parameters.
# Arguments:
#   point (numpy.ndarray): logarithms of (d, alpha, beta, mu, R).
# Returns:
#   numpy.ndarray: logarithms of (d, gamma, mu, R), with gamma = alpha beta.
def _reduce_full(point):
    values = np.asarray(point, dtype=float)  # Full parameter vector in log space.
    advection = values[1] + values[2]  # Logarithm of the product gamma = alpha beta.
    return np.array([values[0], advection, values[3], values[4]])  # Reduced parameter vector.


# Log-likelihood surface over the diffusion rate and the memory decay rate.
# The combined advection strength is held fixed, so that the aggregation ratio
# varies along the surface exactly as Theorem 1(i) describes.
# Arguments:
#   data (dict): the stationary observations and the noise scale.
#   context (dict): the context mapping of the stationary forward map.
#   condition (dict): the condition whose true parameters centre the surface.
#   half_width (float): half width of the surface in natural logarithm units.
#   n_points (int): number of grid points along each axis.
# Returns:
#   tuple: the two axes and the evaluated log-likelihood surface.
def _stationary_surface(data, context, condition, half_width, n_points):
    advection = float(condition["advection"])  # Fixed combined advection strength gamma.
    radius = np.log(float(condition["radius"]))  # Logarithm of the true perceptual range.
    centre_d = np.log(float(condition["diffusion"]))  # Logarithm of the true diffusion rate.
    centre_mu = np.log(float(condition["memory_decay"]))  # Logarithm of the true decay rate.
    axis_d = np.linspace(centre_d - half_width, centre_d + half_width, int(n_points))  # First axis.
    axis_mu = np.linspace(centre_mu - half_width, centre_mu + half_width, int(n_points))  # Second.
    surface = np.full((int(n_points), int(n_points)), -np.inf)  # Storage of the surface values.
    for row, log_d in enumerate(axis_d):  # Vary the diffusion rate along the rows.
        for column, log_mu in enumerate(axis_mu):  # Vary the memory decay rate along the columns.
            ratio = advection / (np.exp(log_d) * np.exp(log_mu))  # Aggregation ratio kappa.
            point = np.array([np.log(ratio), radius])  # Stationary parameters in log space.
            surface[row, column] = forward_stationary_loglik(point, data, context)  # Value.
    return axis_d, axis_mu, surface  # Axes and the evaluated log-likelihood surface.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the experiment and propagate the exit status.
