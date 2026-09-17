# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Comparison of the three candidate models of Section 6 by Pareto
# smoothed importance sampling leave one out cross-validation applied to the
# weighted pointwise contributions of equation (10), and the robustness check
# by leave one block out cross-validation with contiguous blocks of fixes.
# Manuscript: Section 4.5, model comparison; Section 6, Table 8.
# Inputs: matrices of weighted pointwise log-likelihood contributions.
# Outputs: expected log pointwise predictive densities, their differences and
# the rankings implied by the two cross-validation schemes.

import numpy as np  # Numerical arrays and summary statistics.


# Build the ArviZ inference data required by the leave one out estimator.
# ArviZ needs a posterior group in order to read the number of chains and of
# draws, so a placeholder variable of the matching shape is supplied.
# Arguments:
#   pointwise (numpy.ndarray): weighted contributions of shape (draws, fixes).
# Returns:
#   arviz.InferenceData: the log-likelihood group with a placeholder posterior.
def _loo_data(pointwise):
    import arviz  # Imported lazily so that the package imports without ArviZ.

    values = np.asarray(pointwise, dtype=float)  # Weighted contributions of every draw.
    n_draws, n_points = values.shape  # Number of posterior draws and of observations.
    reshaped = values.reshape(1, n_draws, n_points)  # One chain holding all the draws.
    placeholder = {"draw_index": np.arange(n_draws, dtype=float).reshape(1, n_draws)}  # Posterior.
    return arviz.from_dict(posterior=placeholder, log_likelihood={"obs": reshaped})  # Data object.


# Expected log pointwise predictive density of one model by PSIS-LOO.
# Arguments:
#   pointwise (numpy.ndarray): weighted contributions of shape (draws, fixes).
# Returns:
#   dict: the estimate, its standard error and the largest Pareto shape value.
def psis_loo(pointwise):
    import arviz  # Imported lazily so that the package imports without ArviZ.

    data = _loo_data(pointwise)  # Inference data assembled from the weighted contributions.
    result = arviz.loo(data, pointwise=True)  # Pareto smoothed importance sampling estimate.
    shape = result.pareto_k  # Estimated Pareto shape parameter of every observation.
    # Assemble the estimate together with its diagnostics.
    return {
        "elpd": float(result.elpd_loo),  # Expected log pointwise predictive density.
        "se": float(result.se),  # Standard error of the estimate.
        "p_loo": float(result.p_loo),  # Effective number of parameters of the model.
        "max_pareto_k": float(np.max(np.asarray(shape))),  # Largest Pareto shape value.
    }


# Differences in expected log pointwise predictive density between models.
# Arguments:
#   pointwise_by_model (dict): weighted contributions of shape (draws, fixes).
# Returns:
#   list: one record per model with the estimate, the difference from the best
#   model and the standard error of that difference, ordered by the estimate.
def compare_models(pointwise_by_model):
    import arviz  # Imported lazily so that the package imports without ArviZ.

    estimates = {}  # Accumulator for the estimate of every candidate model.
    per_point = {}  # Accumulator for the pointwise estimates of every model.
    for name, values in pointwise_by_model.items():  # Evaluate every candidate model in turn.
        data = _loo_data(values)  # Inference data assembled from the weighted contributions.
        result = arviz.loo(data, pointwise=True)  # Pareto smoothed estimate of the model.
        estimates[name] = result  # Store the full ArviZ result of the model.
        per_point[name] = np.asarray(result.loo_i).ravel()  # Pointwise estimates of the model.
    best = max(estimates, key=lambda name: float(estimates[name].elpd_loo))  # Preferred model.
    records = []  # Accumulator for the comparison table of the individual.
    for name, result in estimates.items():  # Build one row of the comparison table per model.
        difference = float(result.elpd_loo) - float(estimates[best].elpd_loo)  # Difference.
        contrast = per_point[name] - per_point[best]  # Pointwise differences from the best model.
        count = contrast.size  # Number of observations entering the standard error.
        error = float(np.sqrt(count) * np.std(contrast, ddof=1)) if count > 1 else 0.0  # Error.
        # One row of the comparison table, describing the current model.
        record = {
                "model": name,  # Name of the candidate model.
                "elpd": float(result.elpd_loo),  # Expected log pointwise predictive density.
                "se": float(result.se),  # Standard error of the estimate itself.
                "elpd_difference": difference,  # Difference from the preferred model.
                "difference_se": error,  # Standard error of that difference.
                "max_pareto_k": float(np.max(np.asarray(result.pareto_k))),  # Largest shape value.
        }
        records.append(record)  # Store the completed row of the comparison table.
    records.sort(key=lambda record: -record["elpd"])  # Order the table by the estimate.
    return records  # Comparison table of the candidate models of Section 6.


# Assign contiguous blocks of fixed duration to a sequence of timestamps.
# Arguments:
#   times (numpy.ndarray): times of the fixes, in days since the first fix.
#   block_days (float): duration of one block, in days.
# Returns:
#   numpy.ndarray: the integer block index of every fix.
def block_indices(times, block_days=30.0):
    values = np.asarray(times, dtype=float)  # Times of the fixes, in days since the first fix.
    return np.floor((values - float(np.min(values))) / float(block_days)).astype(int)  # Blocks.


# Leave one block out cross-validation of the candidate models.
# Arguments:
#   pointwise_by_model (dict): weighted contributions of shape (draws, fixes).
#   blocks (numpy.ndarray): the integer block index of every fix.
# Returns:
#   list: one record per model with the held out weighted log predictive density.
def leave_one_block_out(pointwise_by_model, blocks):
    indices = np.asarray(blocks, dtype=int)  # Block index of every retained fix.
    labels = np.unique(indices)  # Distinct blocks of the individual.
    records = []  # Accumulator for the cross-validation table of the individual.
    for name, values in pointwise_by_model.items():  # Evaluate every candidate model in turn.
        array = np.asarray(values, dtype=float)  # Weighted contributions of the current model.
        total = 0.0  # Accumulator for the held out log predictive density of the model.
        for label in labels:  # Hold out one contiguous block of fixes at a time.
            selected = indices == label  # Fixes that belong to the held out block.
            draws = np.sum(array[:, selected], axis=1)  # Held out contribution of every draw.
            shifted = draws - float(np.max(draws))  # Shift for a stable logarithm of the mean.
            average = float(np.log(np.mean(np.exp(shifted))) + np.max(draws))  # Log mean density.
            total = total + average  # Accumulate the held out log predictive density.
        records.append({"model": name, "elpd_block": float(total)})  # One record per model.
    records.sort(key=lambda record: -record["elpd_block"])  # Order the table by the estimate.
    return records  # Cross-validation table used as the robustness check of Section 4.5.


# Ranking of the models implied by a comparison table.
# Arguments:
#   records (sequence): the table produced by compare_models or by the block
#   cross-validation.
#   key (str): the column by which the models are ranked.
# Returns:
#   list: the model names in decreasing order of the chosen column.
def ranking(records, key="elpd"):
    ordered = sorted(records, key=lambda record: -float(record[key]))  # Decreasing order.
    return [str(record["model"]) for record in ordered]  # Model names in ranked order.
