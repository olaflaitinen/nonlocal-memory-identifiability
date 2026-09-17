# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Profile likelihood of one parameter, computed in log space with
# repeated local optimisation of the remaining parameters, and the
# classification of the resulting confidence interval.
# Manuscript: Section 4.3, profile likelihood; Experiment 2.3 and Table 6.
# Inputs: a log-likelihood callable, prior bounds and grid settings.
# Outputs: the profile, the classification and the interval bounds.

import numpy as np  # Numerical arrays and elementary functions.
from scipy.optimize import minimize  # Local optimisation with the L-BFGS-B algorithm.

# Threshold of the approximate ninety five per cent confidence interval.
PROFILE_THRESHOLD = 1.9207296556006175  # One half of the chi squared quantile with one degree.


# Profile likelihood of one component of the parameter vector.
# Arguments:
#   loglik (callable): the log-likelihood as a function of the log parameters.
#   bounds (sequence): one pair of log-space bounds per parameter.
#   index (int): index of the parameter that is profiled.
#   n_grid (int): number of equally spaced grid values of the profiled parameter.
#   n_starts (int): number of starting points of the nuisance optimisation.
#   rng (numpy.random.Generator): generator of the random starting points.
#   initial (numpy.ndarray): starting point of the first nuisance optimisation.
#   max_iterations (int): cap on the iterations of one local optimisation, or
#   None for the default of the algorithm. The cap keeps the demonstration
#   configuration inexpensive and is not set in the experiment configurations.
# Returns:
#   dict: the grid of the profiled parameter, the profile values and the
#   optimal nuisance parameters at every grid value.
def profile_parameter(
    loglik,  # Log-likelihood as a function of the parameters in log space.
    bounds,  # One pair of log-space bounds per parameter.
    index,  # Index of the parameter that is profiled.
    n_grid=41,  # Number of equally spaced grid values of the profiled parameter.
    n_starts=8,  # Number of starting points of the nuisance optimisation.
    rng=None,  # Generator of the random starting points.
    initial=None,  # Starting point of the first nuisance optimisation.
    max_iterations=None,  # Cap on the iterations of one local optimisation.
):  # End of the argument list.
    generator = rng if rng is not None else np.random.default_rng(0)  # Deterministic by default.
    box = [(float(low), float(high)) for low, high in bounds]  # Log-space bounds of the parameters.
    options = {"maxiter": int(max_iterations)} if max_iterations else None  # Optimiser options.
    lower = np.array([item[0] for item in box])  # Lower bounds of every parameter, in log space.
    upper = np.array([item[1] for item in box])  # Upper bounds of every parameter, in log space.
    grid = np.linspace(lower[index], upper[index], int(n_grid))  # Grid of the profiled parameter.
    free = [position for position in range(len(box)) if position != index]  # Nuisance positions.
    # Warm start of the first grid value, taken from the caller or from the centre.
    previous = np.array(initial, dtype=float) if initial is not None else 0.5 * (lower + upper)
    profile = np.full(int(n_grid), -np.inf)  # Profile values, one per grid value.
    optima = np.empty((int(n_grid), len(box)), dtype=float)  # Optimal parameters per grid value.
    for position, value in enumerate(grid):  # Optimise the nuisance parameters at each grid value.
        starts = [previous[free]]  # First start is the optimum at the neighbouring grid value.
        for _ in range(max(0, int(n_starts) - 1)):  # Further starts are drawn from the prior box.
            draw = generator.uniform(lower[free], upper[free])  # Log-uniform random start.
            starts.append(draw)  # Store the random starting point.
        best_value = -np.inf  # Largest log-likelihood found at this grid value.
        best_point = previous.copy()  # Parameter vector that attains that log-likelihood.
        for start in starts:  # Run a local optimisation from each starting point.
            # Negative log-likelihood of the nuisance parameters at the fixed grid value.
            def objective(nuisance, value=value, free=free, index=index, size=len(box)):
                point = np.empty(size)  # Full parameter vector assembled from its parts.
                point[index] = value  # Fixed value of the profiled parameter.
                point[free] = nuisance  # Current values of the nuisance parameters.
                result = loglik(point)  # Log-likelihood at the assembled parameter vector.
                return -result if np.isfinite(result) else 1.0e30  # Penalise inadmissible points.

            # Bound constrained local optimisation of the nuisance parameters.
            outcome = minimize(
                objective,  # Negative log-likelihood of the nuisance parameters.
                np.asarray(start, dtype=float),  # Starting point of this local optimisation.
                method="L-BFGS-B",  # Bound constrained quasi-Newton algorithm of Section 4.3.
                bounds=[box[position] for position in free],  # Bounds of the nuisance parameters.
                options=options,  # Optional cap on the iterations of this optimisation.
            )  # Result of one local optimisation of the nuisance parameters.
            candidate = -float(outcome.fun)  # Log-likelihood attained by this local optimum.
            if candidate > best_value:  # This local optimum improves on the previous ones.
                best_value = candidate  # Record the improved log-likelihood.
                best_point = np.empty(len(box))  # Assemble the corresponding parameter vector.
                best_point[index] = value  # Fixed value of the profiled parameter.
                best_point[free] = outcome.x  # Optimal values of the nuisance parameters.
        profile[position] = best_value  # Store the profile value at this grid value.
        optima[position] = best_point  # Store the optimal parameter vector at this grid value.
        previous = best_point  # Warm start of the next grid value, as specified in Section 4.3.
    # Assemble the profile together with the optima that produced it.
    return {
        "grid": grid,  # Grid of the profiled parameter, in log space.
        "profile": profile,  # Profile log-likelihood at every grid value.
        "optima": optima,  # Optimal parameter vector at every grid value.
        "index": int(index),  # Index of the profiled parameter.
    }


# Classify a profile likelihood as practically identifiable or not.
# Arguments:
#   grid (numpy.ndarray): the grid of the profiled parameter, in log space.
#   profile (numpy.ndarray): the profile log-likelihood at the grid values.
#   threshold (float): the threshold Delta of Section 4.3.
# Returns:
#   dict: the classification "I" or "N", the interval bounds in log space and
#   the maximum of the profile.
def classify_interval(grid, profile, threshold=PROFILE_THRESHOLD):
    values = np.asarray(profile, dtype=float)  # Profile log-likelihood at the grid values.
    points = np.asarray(grid, dtype=float)  # Grid of the profiled parameter, in log space.
    finite = np.isfinite(values)  # Grid values at which the profile could be evaluated.
    if not np.any(finite):  # The profile is empty, so no interval can be formed.
        # Without a finite profile value the parameter is not practically identifiable.
        return {"classification": "N", "lower": None, "upper": None, "maximum": None}
    maximum = float(np.max(values[finite]))  # Largest profile value over the grid.
    inside = finite & (maximum - values <= float(threshold))  # Grid values inside the interval.
    if not np.any(inside):  # No grid value lies inside the interval, which cannot happen.
        # An empty interval is reported as not practically identifiable.
        return {"classification": "N", "lower": None, "upper": None, "maximum": maximum}
    first = int(np.argmax(inside))  # Index of the first grid value inside the interval.
    last = len(inside) - 1 - int(np.argmax(inside[::-1]))  # Index of the last such grid value.
    touches_lower = first == 0  # The interval reaches the lower end of the prior range.
    touches_upper = last == len(points) - 1  # The interval reaches the upper end of the range.
    bounded = not (touches_lower or touches_upper)  # A bounded interval lies inside the range.
    # Report the classification together with the bounds of the interval.
    return {
        "classification": "I" if bounded else "N",  # Practically identifiable or not.
        "lower": float(points[first]),  # Lower end of the interval, in log space.
        "upper": float(points[last]),  # Upper end of the interval, in log space.
        "maximum": maximum,  # Largest profile value over the grid.
    }
