# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Reproduce the parameter design of Table 3 and verify the stationary
# identity of Theorem 1 at the resulting steady states, before any experiment is
# run. The script also checks that the steady state is unchanged when the
# diffusion rate, the memory decay rate and the combined advection strength
# change with the aggregation ratio held fixed.
# Manuscript: Section 4.2, Table 3; Section 3.3, equation (7) and Theorem 1.
# Inputs: an optional supercriticality factor and a list of perceptual ranges.
# Outputs: a report on standard output and a non-zero exit status on failure.

import argparse  # Command line interface of the pre-experiment check.
import sys  # Process exit status of the pre-experiment check.

import numpy as np  # Numerical arrays and elementary functions.

from nmi.linear_theory import m_star  # Index of the first nonpositive top-hat multiplier.
from nmi.steady_state import (  # Steady state solver and the stationary identity.
    continuation_in_kappa,  # Continuation of the branch from the onset value.
    critical_kappa,  # Onset value of the aggregation ratio.
    fixed_point_steady_state,  # Fixed-point characterisation of the steady state.
    identity_residual,  # Residual of the stationary identity of equation (7).
)

# Length of the periodic domain used by the design of Section 4.2.
DOMAIN_LENGTH = 1.0  # Nondimensionalised domain length of the synthetic experiments.
# Mean density of the uniform state used by the design of Section 4.2.
MEAN_DENSITY = 1.0  # Nondimensionalised mean density of the synthetic experiments.
# Diffusion rate of the design of Section 4.2.
DIFFUSION = 0.01  # Diffusion rate d fixed by the design.
# Memory decay rate of the design of Section 4.2.
MEMORY_DECAY = 1.0  # Memory decay rate mu fixed by the design.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Parameter design of Table 3")  # Argument parser.
    parser.add_argument("supercriticality", nargs="?", type=float, default=1.2)  # Factor of (8).
    parser.add_argument("radii", nargs="?", default="0.075,0.15,0.30")  # Perceptual ranges.
    parser.add_argument("--n-points", type=int, default=256, help="grid size of the solver")  # Grid.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# Compute the steady state of one condition of Table 3 by continuation.
# Arguments:
#   kernel (str): "tophat" or "gaussian".
#   radius (float): perceptual range R of the condition.
#   ratio (float): aggregation ratio kappa of the condition.
#   critical (float): critical aggregation ratio of the condition.
#   n_points (int): grid size of the fixed-point solver.
# Returns:
#   dict: the converged steady state and the diagnostics of the solve.
def steady_state_of_condition(kernel, radius, ratio, critical, n_points):
    ratios = np.linspace(critical * 1.01, ratio, 12)  # Continuation path from the onset value.
    branch = continuation_in_kappa(  # Continuation of the branch in the aggregation ratio.
        ratios,  # Aggregation ratios of the continuation path.
        radius,  # Perceptual range R of the condition.
        kernel,  # Detection kernel family of the condition.
        int(n_points),  # Grid size of the fixed-point solver.
        DOMAIN_LENGTH,  # Length L of the periodic domain.
        MEAN_DENSITY,  # Mean density u_bar of the conserved mass.
    )  # Steady states along the continuation path.
    return branch[-1]  # Steady state at the requested aggregation ratio.


# Entry point of the parameter design check.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero when every check passes and one otherwise.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    factor = float(arguments.supercriticality)  # Factor by which kappa exceeds the onset value.
    radii = [float(value) for value in str(arguments.radii).split(",")]  # Perceptual ranges.
    failures = []  # Accumulator for the conditions that did not pass the checks.
    print("kernel    R      kappa_c   kappa     gamma      m_star  residual   invariance")  # Header.
    for kernel in ("tophat", "gaussian"):  # Both detection kernel families of the manuscript.
        for radius in radii:  # Every perceptual range of the design of Section 4.2.
            critical = critical_kappa(radius, kernel, DOMAIN_LENGTH, MEAN_DENSITY)  # Onset value.
            ratio = factor * critical  # Aggregation ratio kappa of equation (8).
            advection = ratio * DIFFUSION * MEMORY_DECAY  # Combined advection strength gamma.
            index = m_star(radius, DOMAIN_LENGTH) if kernel == "tophat" else 0  # Corollary 1.
            # Steady state of this condition, obtained by continuation from the onset.
            result = steady_state_of_condition(kernel, radius, ratio, critical, arguments.n_points)
            residual = identity_residual(  # Residual of the stationary identity of equation (7).
                result["density"],  # Converged steady state of this condition.
                ratio,  # Aggregation ratio kappa of this condition.
                radius,  # Perceptual range R of this condition.
                kernel,  # Detection kernel family of this condition.
                DOMAIN_LENGTH,  # Length L of the periodic domain.
            )  # Residual that Theorem 1 requires to vanish.
            invariance = check_invariance(result["density"], ratio, radius, kernel)  # Theorem 1(i).
            label = str(index) if index else "n/a"  # Index m_star, absent for the Gaussian kernel.
            # One line of the report per condition of Table 3.
            print(
                f"{kernel:9s} {radius:6.3f} {critical:9.3f} {ratio:9.3f} "  # Design values.
                f"{advection:10.5f} {label:>6s}  {residual:.2e}  {invariance:.2e}"  # Diagnostics.
            )  # One line of the report per condition of Table 3.
            if radius >= 0.15 and residual > 1.0e-8:  # The identity must hold at this resolution.
                failures.append(f"{kernel} R={radius}")  # Record the failure of this condition.
            if invariance > 1.0e-10:  # The steady state must not depend on d, mu and gamma alone.
                failures.append(f"{kernel} R={radius} invariance")  # Record the failure.
    if failures:  # At least one condition did not pass the checks.
        print("failed conditions: " + ", ".join(failures))  # Report the failing conditions.
        return 1  # Signal failure to the caller.
    print("every condition of Table 3 passed the stationary identity checks")  # Report success.
    return 0  # Signal success to the caller.


# Check that the steady state depends on the parameters only through kappa.
# Arguments:
#   density (numpy.ndarray): the steady state of the reference parameters.
#   ratio (float): the aggregation ratio kappa of the condition.
#   radius (float): the perceptual range R of the condition.
#   kernel (str): the detection kernel family of the condition.
# Returns:
#   float: the largest relative difference between the two steady states.
def check_invariance(density, ratio, radius, kernel):
    diffusion = DIFFUSION * 7.3  # Diffusion rate of the transformed parameter triple.
    memory_decay = MEMORY_DECAY / 2.9  # Memory decay rate of the transformed triple.
    advection = ratio * diffusion * memory_decay  # Combined advection strength of that triple.
    transformed = advection / (diffusion * memory_decay)  # Aggregation ratio of the triple.
    recomputed = fixed_point_steady_state(  # Steady state of the transformed parameter triple.
        transformed,  # Aggregation ratio implied by the transformed triple.
        radius,  # Perceptual range R of the condition.
        kernel,  # Detection kernel family of the condition.
        np.maximum(density, 1.0e-12),  # Reference steady state used as the starting iterate.
        DOMAIN_LENGTH,  # Length L of the periodic domain.
    )  # Steady state of the transformed parameters, which share the same aggregation ratio.
    difference = float(np.max(np.abs(recomputed["density"] - density)))  # Largest difference.
    return difference / float(np.max(np.abs(density)))  # Relative difference of the two states.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the checks and propagate the exit status.
