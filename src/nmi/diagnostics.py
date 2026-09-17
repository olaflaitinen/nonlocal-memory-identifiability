# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Convergence diagnostics of the sampled posteriors. The module
# computes the rank-normalised split R-hat and the bulk effective sample size
# with ArviZ, applies the acceptance criteria of Section 4.4 and lists the
# conditions that must be rerun.
# Manuscript: Section 4.4, Bayesian inference; Experiment 2.5.
# Inputs: chains of one or several conditions. Outputs: diagnostic tables and
# the list of conditions that failed the criteria.

import numpy as np  # Numerical arrays and summary statistics.

# Threshold on the rank-normalised split R-hat required by Section 4.4.
RHAT_THRESHOLD = 1.01  # A condition converges only when every parameter is below this value.
# Threshold on the bulk effective sample size required by Section 4.4.
ESS_THRESHOLD = 400.0  # A condition converges only when every parameter reaches this value.


# Build an ArviZ inference data object from an array of chains.
# Arguments:
#   chains (numpy.ndarray): array of shape (n_chains, n_draws, n_parameters).
#   names (sequence): names of the sampled parameters, in the same order.
# Returns:
#   arviz.InferenceData: the posterior group holding the supplied chains.
def to_inference_data(chains, names):
    import arviz  # Imported lazily so that the package imports without ArviZ.

    array = np.asarray(chains, dtype=float)  # Chains of shape (chains, draws, parameters).
    posterior = {name: array[:, :, index] for index, name in enumerate(names)}  # One key per name.
    return arviz.from_dict(posterior=posterior)  # Inference data used by the ArviZ diagnostics.


# Rank-normalised split R-hat and bulk effective sample size of every parameter.
# Arguments:
#   chains (numpy.ndarray): array of shape (n_chains, n_draws, n_parameters).
#   names (sequence): names of the sampled parameters, in the same order.
# Returns:
#   dict: one entry per parameter holding its R-hat and effective sample size.
def convergence_statistics(chains, names):
    import arviz  # Imported lazily so that the package imports without ArviZ.

    data = to_inference_data(chains, names)  # Inference data built from the supplied chains.
    rhat = arviz.rhat(data, method="rank")  # Rank-normalised split R-hat of every parameter.
    ess = arviz.ess(data, method="bulk")  # Bulk effective sample size of every parameter.
    statistics = {}  # Accumulator for the diagnostics of every parameter.
    for name in names:  # Record the two diagnostics for each sampled parameter.
        # Both diagnostics of the current parameter, stored under its name.
        statistics[name] = {
            "rhat": float(rhat[name].values),  # Rank-normalised split R-hat of the parameter.
            "ess_bulk": float(ess[name].values),  # Bulk effective sample size of the parameter.
        }
    return statistics  # Diagnostics of every sampled parameter of the condition.


# Decide whether a condition satisfies the convergence criteria of Section 4.4.
# Arguments:
#   statistics (dict): the mapping produced by convergence_statistics.
# Returns:
#   bool: True when every parameter meets both thresholds.
def has_converged(statistics):
    for entry in statistics.values():  # Inspect the diagnostics of every parameter.
        if not np.isfinite(entry["rhat"]) or entry["rhat"] >= RHAT_THRESHOLD:  # R-hat too large.
            return False  # The condition fails the convergence criteria.
        if entry["ess_bulk"] < ESS_THRESHOLD:  # The effective sample size is too small.
            return False  # The condition fails the convergence criteria.
    return True  # Every parameter satisfies both criteria of Section 4.4.


# Posterior summaries of one condition, as reported in Section 4.4.
# Arguments:
#   chains (numpy.ndarray): array of shape (n_chains, n_draws, n_parameters).
#   names (sequence): names of the sampled parameters, in the same order.
#   truth (dict): the true value of each parameter, used for the relative width.
# Returns:
#   dict: one entry per parameter with the median, the credible interval bounds
#   and the relative interval width w_r of Section 4.4.
def posterior_summary(chains, names, truth):
    array = np.asarray(chains, dtype=float)  # Chains of shape (chains, draws, parameters).
    flattened = array.reshape(-1, array.shape[-1])  # Pooled draws of every chain.
    summary = {}  # Accumulator for the summaries of every parameter.
    for index, name in enumerate(names):  # Summarise each sampled parameter in turn.
        draws = flattened[:, index]  # Pooled draws of the current parameter.
        lower, median, upper = np.quantile(draws, [0.025, 0.5, 0.975])  # Credible interval.
        true_value = float(truth[name])  # True value of the parameter in this condition.
        # Posterior summary of the current parameter, stored under its name.
        summary[name] = {
            "median": float(median),  # Posterior median of the parameter.
            "lower": float(lower),  # Lower end of the ninety five per cent credible interval.
            "upper": float(upper),  # Upper end of the ninety five per cent credible interval.
            "relative_width": float((upper - lower) / true_value),  # Relative width w_r.
            "truth": true_value,  # True value recorded alongside the summary.
        }
    return summary  # Posterior summaries of every sampled parameter.


# Build the table of conditions that must be rerun.
# Arguments:
#   records (sequence): dictionaries with keys "identifier" and "converged".
# Returns:
#   list: the identifiers of the conditions that failed the criteria.
def rerun_list(records):
    return [str(record["identifier"]) for record in records if not record["converged"]]  # Failures.
