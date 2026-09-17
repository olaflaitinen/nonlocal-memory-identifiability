# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Numerical verification of the structural results of Section 3
# before any experiment is run. The script checks the scaling symmetry of
# Proposition 1, the instability criterion of Proposition 3, the injectivity of
# Lemma 1, the parameter reconstruction of Theorem 2 and the sign property of
# Corollary 1.
# Manuscript: Section 3, Propositions 1 and 3, Lemma 1, Theorem 2, Corollary 1.
# Inputs: optional command line arguments that set the sample sizes.
# Outputs: a report on standard output and a non-zero exit status on failure.

import argparse  # Command line interface of the pre-experiment check.
import sys  # Process exit status of the pre-experiment check.

import numpy as np  # Numerical arrays and random number generation.

from nmi.kernels import kernel_hat_1d  # Fourier transforms of the detection kernels.
from nmi.linear_theory import (  # Linear theory and the reconstruction of Theorem 2.
    is_unstable,  # Instability criterion of Proposition 3.
    m_star,  # Index of the first nonpositive top-hat multiplier.
    mode_matrix,  # Mode matrix of equation (6).
    mode_trajectory,  # Exact trajectory of one linearised Fourier mode.
    reconstruct_parameters_theorem2,  # Parameter reconstruction of Theorem 2.
)
from nmi.spectral_1d import simulate_1d  # Pseudo-spectral forward solver.

# Length of the periodic domain used by every check of this script.
DOMAIN_LENGTH = 1.0  # Nondimensionalised domain length of Section 4.2.
# Mean density of the uniform state used by every check of this script.
MEAN_DENSITY = 1.0  # Nondimensionalised mean density of Section 4.2.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Pre-experiment proof checks")  # Argument parser.
    parser.add_argument("--n-stability", type=int, default=20000, help="stability draws")  # Size.
    parser.add_argument("--n-recovery", type=int, default=400, help="recovery draws")  # Size.
    parser.add_argument("--n-grid", type=int, default=20000, help="grid size")  # Grid resolution.
    parser.add_argument("--seed", type=int, default=20260101, help="seed of the draws")  # Seed.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# Check the instability criterion of Proposition 3 against direct eigenvalues.
# Arguments:
#   n_draws (int): number of random parameter sets inspected.
#   rng (numpy.random.Generator): generator of the random parameter sets.
# Returns:
#   tuple: the number of agreements and the number of parameter sets inspected.
def check_proposition_3(n_draws, rng):
    agreements = 0  # Number of parameter sets at which the criterion agrees.
    for _ in range(int(n_draws)):  # Inspect one random parameter set at a time.
        diffusion = 10.0 ** rng.uniform(-3.0, -1.0)  # Diffusion rate d of this parameter set.
        memory_decay = 10.0 ** rng.uniform(-1.0, 1.0)  # Memory decay rate mu of this set.
        advection = 10.0 ** rng.uniform(-4.0, 0.0)  # Combined advection strength gamma.
        radius = rng.uniform(0.02, 0.48)  # Perceptual range R of this parameter set.
        kernel = "tophat" if rng.uniform() < 0.5 else "gaussian"  # Detection kernel family.
        ratio = advection / (diffusion * memory_decay)  # Aggregation ratio kappa of this set.
        # Verdict of the criterion of Proposition 3 at this parameter set.
        predicted = is_unstable(ratio, radius, kernel, DOMAIN_LENGTH, MEAN_DENSITY, m_max=64)
        observed = False  # Whether a direct eigenvalue computation finds an unstable mode.
        for mode in range(1, 65):  # Inspect the same modes as the criterion does.
            wavenumber = 2.0 * np.pi * mode / DOMAIN_LENGTH  # Wavenumber xi_m of this mode.
            matrix = mode_matrix(  # Mode matrix of equation (6) at this wavenumber.
                wavenumber,  # Wavenumber of the current mode.
                diffusion,  # Diffusion rate d of this parameter set.
                advection,  # Advection strength alpha, with the uptake rate set to one.
                1.0,  # Memory uptake rate beta, set to one without loss of generality.
                memory_decay,  # Memory decay rate mu of this parameter set.
                radius,  # Perceptual range R of this parameter set.
                kernel,  # Detection kernel family of this parameter set.
                MEAN_DENSITY,  # Mean density u_bar of the uniform state.
            )  # Mode matrix whose spectrum is computed directly.
            if float(np.max(np.real(np.linalg.eigvals(matrix)))) > 0.0:  # An unstable eigenvalue.
                observed = True  # The direct computation finds an unstable mode.
                break  # No further mode needs to be inspected.
        agreements += int(predicted == observed)  # Count the agreement of the two verdicts.
    return agreements, int(n_draws)  # Agreements and the number of parameter sets inspected.


# Check the injectivity stated in Lemma 1 on a fine grid of angles.
# Arguments:
#   n_grid (int): number of angles inspected.
# Returns:
#   float: the smallest separation between two distinct angles that agree.
def check_lemma_1(n_grid):
    angles = np.linspace(1.0e-6, np.pi - 1.0e-6, int(n_grid))  # Angles theta in the open interval.
    cosines = np.cos(angles)  # The cosine, which Lemma 1 requires to be injective.
    differences = np.diff(cosines)  # Differences between the cosines of neighbouring angles.
    return float(np.max(differences))  # Largest difference, which must be strictly negative.


# Check the sign property of Corollary 1 on a fine grid of perceptual ranges.
# Arguments:
#   n_grid (int): number of perceptual ranges inspected.
# Returns:
#   tuple: the largest observed value of sin(m_star theta) and the grid size.
def check_corollary_1(n_grid):
    radii = np.linspace(1.0e-4, 0.5 * DOMAIN_LENGTH - 1.0e-6, int(n_grid))  # Perceptual ranges.
    largest = -np.inf  # Largest observed value of the sine, which must not be positive.
    for radius in radii:  # Inspect one perceptual range at a time.
        angle = 2.0 * np.pi * radius / DOMAIN_LENGTH  # Angle theta of this perceptual range.
        index = m_star(float(radius), DOMAIN_LENGTH)  # Index m_star of Corollary 1.
        largest = max(largest, float(np.sin(index * angle)))  # Track the largest observed sine.
    return largest, int(n_grid)  # Largest observed sine and the number of ranges inspected.


# Check the parameter reconstruction of Theorem 2 on random parameter sets.
# Arguments:
#   n_draws (int): number of random parameter sets inspected.
#   rng (numpy.random.Generator): generator of the random parameter sets.
# Returns:
#   tuple: the median relative error, the largest relative error and the count.
def check_theorem_2(n_draws, rng):
    errors = []  # Relative errors of the reconstruction, one per admissible parameter set.
    wavenumber_2 = 4.0 * np.pi / DOMAIN_LENGTH  # Wavenumber xi_2 of the second observed mode.
    times = np.linspace(0.0, 2.0, 40)  # Times at which the modes are observed.
    while len(errors) < int(n_draws):  # Continue until enough admissible sets were inspected.
        diffusion = 10.0 ** rng.uniform(-3.0, -1.0)  # Diffusion rate d of this parameter set.
        memory_decay = 10.0 ** rng.uniform(-1.0, 1.0)  # Memory decay rate mu of this set.
        advection = 10.0 ** rng.uniform(-3.0, -1.0)  # Combined advection strength gamma.
        radius = rng.uniform(0.02, 0.48)  # Perceptual range R of this parameter set.
        kernel = "tophat" if rng.uniform() < 0.5 else "gaussian"  # Detection kernel family.
        if abs(float(kernel_hat_1d(radius * wavenumber_2, kernel))) < 1.0e-2:  # Degenerate mode.
            continue  # The second mode carries no information at this parameter set.
        trajectories = []  # Exact trajectories of the first two linearised Fourier modes.
        for mode in (1, 2):  # Observe the first two modes, as Theorem 2 requires.
            wavenumber = 2.0 * np.pi * mode / DOMAIN_LENGTH  # Wavenumber of this mode.
            matrix = mode_matrix(  # Mode matrix of equation (6) at this wavenumber.
                wavenumber,  # Wavenumber of the current mode.
                diffusion,  # Diffusion rate d of this parameter set.
                advection,  # Advection strength alpha, with the uptake rate set to one.
                1.0,  # Memory uptake rate beta, set to one without loss of generality.
                memory_decay,  # Memory decay rate mu of this parameter set.
                radius,  # Perceptual range R of this parameter set.
                kernel,  # Detection kernel family of this parameter set.
                MEAN_DENSITY,  # Mean density u_bar of the uniform state.
            )  # Mode matrix whose exact trajectory is evaluated.
            trajectories.append(mode_trajectory(matrix, np.array([1.0, 0.0]), times))  # Trajectory.
        recovered = reconstruct_parameters_theorem2(  # Reconstruction of Theorem 2.
            trajectories[0],  # Observed trajectory of the first mode.
            trajectories[1],  # Observed trajectory of the second mode.
            DOMAIN_LENGTH,  # Length L of the periodic domain.
            MEAN_DENSITY,  # Mean density u_bar of the uniform state.
            kernel,  # Detection kernel family, assumed known.
        )  # Recovered parameters of this parameter set.
        error = max(  # Largest relative error over the four recovered parameters.
            abs(recovered["diffusion"] - diffusion) / diffusion,  # Diffusion rate d.
            abs(recovered["memory_decay"] - memory_decay) / memory_decay,  # Memory decay rate mu.
            abs(recovered["advection"] - advection) / advection,  # Combined advection strength.
            abs(recovered["radius"] - radius) / radius,  # Perceptual range R.
        )  # Relative error of this parameter set.
        errors.append(error)  # Store the relative error of this parameter set.
    return float(np.median(errors)), float(np.max(errors)), len(errors)  # Errors and the count.


# Check the scaling symmetry of Proposition 1 in the nonlinear solver.
# Arguments:
#   none.
# Returns:
#   float: the largest relative difference between the two transformed solutions.
def check_proposition_1():
    n_points = 128  # Grid size of the two forward solves compared by this check.
    grid = np.arange(n_points) * (DOMAIN_LENGTH / n_points)  # Grid points of the domain.
    initial = MEAN_DENSITY * (1.0 + 0.05 * np.cos(2.0 * np.pi * grid / DOMAIN_LENGTH))  # Data.
    base = {  # Model parameters of the reference solve.
        "diffusion": 0.01,  # Diffusion rate d of the reference solve.
        "alpha": 0.02,  # Advection strength alpha of the reference solve.
        "beta": 1.0,  # Memory uptake rate beta of the reference solve.
        "memory_decay": 1.0,  # Memory decay rate mu of the reference solve.
        "radius": 0.15,  # Perceptual range R of the reference solve.
    }
    scale = 3.7  # Factor c of the scaling symmetry of Proposition 1.
    scaled = dict(base)  # Model parameters of the transformed solve.
    scaled["alpha"] = base["alpha"] / scale  # Advection strength divided by the factor c.
    scaled["beta"] = base["beta"] * scale  # Memory uptake rate multiplied by the factor c.
    # Forward solve with the reference parameters of the symmetry check.
    first = simulate_1d(base, "tophat", initial, 0.0, 5.0, 0.005, n_points, DOMAIN_LENGTH)
    # Forward solve with the transformed parameters of Proposition 1.
    second = simulate_1d(scaled, "tophat", initial, 0.0, 5.0, 0.005, n_points, DOMAIN_LENGTH)
    difference = np.max(np.abs(first["final_density"] - second["final_density"]))  # Difference.
    return float(difference / np.max(np.abs(first["final_density"])))  # Relative difference.


# Entry point of the pre-experiment proof checks.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero when every check passes and one otherwise.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    rng = np.random.default_rng(int(arguments.seed))  # Generator of the random parameter sets.
    failures = []  # Accumulator for the checks that did not pass.
    symmetry = check_proposition_1()  # Scaling symmetry of Proposition 1.
    print(f"Proposition 1: relative difference {symmetry:.3e}")  # Report the observed difference.
    if symmetry > 1.0e-12:  # The symmetry must hold to machine precision.
        failures.append("Proposition 1")  # Record the failure of this check.
    agreements, inspected = check_proposition_3(arguments.n_stability, rng)  # Proposition 3.
    # Report how often the criterion and the direct eigenvalue computation agreed.
    print(f"Proposition 3: criterion agreed in {agreements} of {inspected} parameter sets")
    if agreements != inspected:  # The criterion must agree in every inspected parameter set.
        failures.append("Proposition 3")  # Record the failure of this check.
    separation = check_lemma_1(arguments.n_grid)  # Injectivity of the cosine of Lemma 1.
    # Report the observed monotonicity, which must be strictly decreasing.
    print(f"Lemma 1: largest difference between neighbouring cosines {separation:.3e}")
    if separation >= 0.0:  # The cosine must decrease strictly on the inspected interval.
        failures.append("Lemma 1")  # Record the failure of this check.
    median, largest, count = check_theorem_2(arguments.n_recovery, rng)  # Theorem 2.
    # Report the accuracy of the reconstruction over the inspected parameter sets.
    print(f"Theorem 2: median relative error {median:.3e}, largest {largest:.3e} over {count} cases")
    if largest > 1.0e-3:  # The reconstruction must be accurate at every admissible parameter set.
        failures.append("Theorem 2")  # Record the failure of this check.
    sine, grid_size = check_corollary_1(arguments.n_grid)  # Sign property of Corollary 1.
    # Report the largest observed sine, which must be nonpositive up to rounding.
    print(f"Corollary 1: largest observed sine {sine:.3e} over {grid_size} perceptual ranges")
    if sine > 1.0e-12:  # The sine must be nonpositive at the index m_star.
        failures.append("Corollary 1")  # Record the failure of this check.
    if failures:  # At least one structural check did not pass.
        print("failed checks: " + ", ".join(failures))  # Report the failing checks.
        return 1  # Signal failure to the caller.
    print("every structural check passed")  # Report a successful run of the script.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the checks and propagate the exit status.
