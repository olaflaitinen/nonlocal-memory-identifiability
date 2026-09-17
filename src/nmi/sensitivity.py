# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Local sensitivity analysis of the forward map. The module builds the
# finite-difference sensitivity matrix with respect to the logarithms of the
# parameters, reports its singular values and forms the Fisher information
# matrix of the Gaussian observation model together with its condition number.
# Manuscript: Section 5.2, Experiment 2.1.
# Inputs: a prediction callable and a parameter vector in log space.
# Outputs: sensitivity matrices, singular values and information matrices.

import numpy as np  # Numerical arrays and linear algebra.


# Finite-difference sensitivity matrix with respect to the log parameters.
# Arguments:
#   predict (callable): maps parameters in log space to a prediction array.
#   theta_log (numpy.ndarray): the point at which the derivative is taken.
#   step (float): relative step of the central difference in log space.
# Returns:
#   numpy.ndarray: the matrix of shape (number of observations, number of
#   parameters) whose columns are the derivatives with respect to each log
#   parameter.
def sensitivity_matrix(predict, theta_log, step=1.0e-4):
    base = np.asarray(theta_log, dtype=float)  # Point at which the derivative is taken.
    reference = np.asarray(predict(base), dtype=float).ravel()  # Prediction at that point.
    columns = np.empty((reference.size, base.size), dtype=float)  # Storage of the derivatives.
    for index in range(base.size):  # Differentiate with respect to each log parameter in turn.
        forward = base.copy()  # Copy of the point that is perturbed upwards.
        backward = base.copy()  # Copy of the point that is perturbed downwards.
        forward[index] = forward[index] + step  # Upward perturbation of the current parameter.
        backward[index] = backward[index] - step  # Downward perturbation of the same parameter.
        upper = np.asarray(predict(forward), dtype=float).ravel()  # Prediction after the step up.
        lower = np.asarray(predict(backward), dtype=float).ravel()  # Prediction after the step down.
        columns[:, index] = (upper - lower) / (2.0 * step)  # Central difference approximation.
    return columns  # Sensitivity matrix with respect to the log parameters.


# Singular values of a sensitivity matrix.
# Arguments:
#   matrix (numpy.ndarray): the sensitivity matrix.
# Returns:
#   numpy.ndarray: the singular values in decreasing order.
def singular_values(matrix):
    return np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)  # Decreasing order.


# Fisher information matrix of the Gaussian observation model (9).
# Arguments:
#   matrix (numpy.ndarray): the sensitivity matrix with respect to log parameters.
#   sigma (float): the common standard deviation of the observation noise.
# Returns:
#   numpy.ndarray: the Fisher information matrix in the log parameters.
def fisher_information(matrix, sigma):
    jacobian = np.asarray(matrix, dtype=float)  # Sensitivity matrix of the forward map.
    return jacobian.T @ jacobian / float(sigma) ** 2  # Information matrix of a Gaussian model.


# Condition number of a symmetric positive semidefinite matrix.
# Arguments:
#   matrix (numpy.ndarray): the matrix whose condition number is required.
# Returns:
#   float: the ratio of the largest to the smallest eigenvalue, or infinity.
def condition_number(matrix):
    eigenvalues = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))  # Real eigenvalues.
    smallest = float(np.min(eigenvalues))  # Smallest eigenvalue of the information matrix.
    largest = float(np.max(eigenvalues))  # Largest eigenvalue of the information matrix.
    if smallest <= 0.0:  # A rank deficient matrix has an unbounded condition number.
        return float("inf")  # Report an infinite condition number for a singular matrix.
    return largest / smallest  # Condition number of the information matrix.


# Numerical rank of a matrix at a relative tolerance on its singular values.
# Arguments:
#   values (numpy.ndarray): the singular values in decreasing order.
#   tolerance (float): relative tolerance applied to the largest singular value.
# Returns:
#   int: the number of singular values above the tolerance.
def numerical_rank(values, tolerance=1.0e-8):
    array = np.asarray(values, dtype=float)  # Singular values in decreasing order.
    if array.size == 0 or array[0] <= 0.0:  # A zero matrix has rank zero by convention.
        return 0  # The matrix carries no information about any parameter.
    return int(np.sum(array > tolerance * array[0]))  # Count of the significant singular values.
