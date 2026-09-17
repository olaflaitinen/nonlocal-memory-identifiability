# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Shared fixtures of the test suite, providing the design
# configuration of the manuscript and the constants of Section 4.2.
# Manuscript: supports the verification described in docs/reproducibility.md.
# Inputs: the tracked configuration files. Outputs: pytest fixtures.

from pathlib import Path  # Portable filesystem paths.

import pytest  # Fixture declaration of the test runner.

from nmi.config import load_experiment_config  # Configuration loader with base inheritance.

# Length of the periodic domain used throughout the synthetic experiments.
DOMAIN_LENGTH = 1.0  # Nondimensionalised domain length of Section 4.2.
# Mean density of the uniform state used throughout the synthetic experiments.
MEAN_DENSITY = 1.0  # Nondimensionalised mean density of Section 4.2.


# Path of the repository root, derived from the location of this file.
# Arguments:
#   none.
# Returns:
#   pathlib.Path: the root directory of the repository.
@pytest.fixture(scope="session")
def repository_root():  # Root directory fixture of the test session.
    return Path(__file__).resolve().parents[1]  # Parent directory of the tests directory.


# Design configuration of the manuscript, loaded once per test session.
# Arguments:
#   repository_root (pathlib.Path): the root directory of the repository.
# Returns:
#   dict: the merged and validated design configuration.
@pytest.fixture(scope="session")
def design_config(repository_root):  # Design configuration fixture.
    return load_experiment_config(repository_root / "configs" / "design_1d.yaml")  # Design of the paper.


# Smoke configuration, loaded once per test session.
# Arguments:
#   repository_root (pathlib.Path): the root directory of the repository.
# Returns:
#   dict: the merged and validated smoke configuration.
@pytest.fixture(scope="session")
def smoke_config(repository_root):  # Smoke configuration fixture.
    return load_experiment_config(repository_root / "configs" / "smoke.yaml")  # Smoke settings.
