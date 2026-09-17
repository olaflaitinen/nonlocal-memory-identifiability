# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Loading, merging and validation of the YAML configuration files that
# define the numerical resolutions, the synthetic design and the sampler
# settings of every experiment.
# Manuscript: Sections 4.1 to 4.5; the configuration files are listed in
# docs/experiments.md.
# Inputs: paths of YAML files. Outputs: nested dictionaries of configuration
# values, together with a stable hash used to label experiment output.

import copy  # Deep copies that keep the caller's configuration unchanged.
import hashlib  # Stable hashing of configurations for the output metadata.
import json  # Canonical serialisation used as the input of the hash.
from pathlib import Path  # Portable filesystem paths.

import yaml  # Parser for the YAML configuration files.

# Keys that every experiment configuration must define at the top level.
REQUIRED_TOP_LEVEL = ("experiment", "design", "solver")  # Minimal configuration contract.

# Keys that the design block must define.
REQUIRED_DESIGN = (
    "domain_length",  # Length L of the periodic domain.
    "mean_density",  # Mean density u_bar of the conserved mass.
    "beta",  # Memory uptake rate, fixed to one by Proposition 1.
    "diffusion",  # Diffusion rate d of the density equation.
    "memory_decay",  # Memory decay rate mu of the map equation.
    "perturbation_amplitude",  # Amplitude epsilon of the initial perturbation.
    "observation_time",  # Upper end T of the observation window.
    "supercriticality",  # Factor by which kappa exceeds the onset value.
    "perceptual_ranges",  # Perceptual ranges R of the factorial design.
    "kernels",  # Detection kernel families of the factorial design.
    "noise_levels",  # Relative noise levels eta of the factorial design.
    "sampling_designs",  # Named sampling designs of the factorial design.
    "seed",  # Global seed from which every per-condition seed is spawned.
)  # End of the required design keys.

# Keys that the solver block must define.
REQUIRED_SOLVER = (
    "inference",  # Resolution used when evaluating the likelihood.
    "reference",  # Resolution used when generating synthetic data.
    "scheme",  # Time stepping scheme, "imex_euler" or "sbdf2".
    "positivity_tolerance",  # Relative tolerance below which a run is rejected.
)  # End of the required solver keys.


# Read a YAML configuration file from disk.
# Arguments:
#   path (str or pathlib.Path): path of the YAML file to read.
# Returns:
#   dict: the parsed configuration, with an added "config_path" entry.
def load_config(path):
    location = Path(path)  # Normalise the argument into a path object.
    if not location.is_file():  # The configuration file does not exist.
        raise FileNotFoundError(f"Configuration file not found: {location}")  # Report clearly.
    text = location.read_text(encoding="utf-8")  # Read the whole file as text.
    parsed = yaml.safe_load(text)  # Parse the YAML document without executing tags.
    if not isinstance(parsed, dict):  # A configuration must be a mapping at the top level.
        raise ValueError(f"Configuration {location} does not contain a top level mapping")  # Reject.
    parsed["config_path"] = str(location)  # Record the origin of the configuration.
    validate_config(parsed)  # Apply the schema checks before returning the configuration.
    return parsed  # Return the validated configuration mapping.


# Merge an override mapping into a base mapping, recursively.
# Arguments:
#   base (dict): the configuration that supplies the default values.
#   override (dict): the configuration whose entries take precedence.
# Returns:
#   dict: a new mapping in which the override entries replace the base entries.
def merge_configs(base, override):
    merged = copy.deepcopy(base)  # Work on a copy so that the caller is unaffected.
    for key, value in override.items():  # Consider every entry of the override mapping.
        # Nested mappings are merged entry by entry rather than replaced wholesale.
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)  # Recurse into nested mappings.
        else:  # Scalars, sequences and new keys replace the base entry outright.
            merged[key] = copy.deepcopy(value)  # Store an independent copy of the value.
    return merged  # Return the merged configuration mapping.


# Validate that a configuration defines the keys required by the experiments.
# Arguments:
#   config (dict): the configuration mapping to validate.
# Returns:
#   None: the function returns quietly or raises ValueError with a clear message.
def validate_config(config):
    for key in REQUIRED_TOP_LEVEL:  # Check the blocks required at the top level.
        if key not in config:  # The block is absent from the configuration.
            raise ValueError(f"Configuration is missing the required block {key!r}")  # Reject.
    design = config["design"]  # Block that describes the synthetic design.
    for key in REQUIRED_DESIGN:  # Check the entries required inside the design block.
        if key not in design:  # The entry is absent from the design block.
            raise ValueError(f"Configuration block 'design' is missing the key {key!r}")  # Reject.
    solver = config["solver"]  # Block that describes the numerical resolutions.
    for key in REQUIRED_SOLVER:  # Check the entries required inside the solver block.
        if key not in solver:  # The entry is absent from the solver block.
            raise ValueError(f"Configuration block 'solver' is missing the key {key!r}")  # Reject.
    for name in ("inference", "reference"):  # Both resolutions must define a grid and a step.
        resolution = solver[name]  # Resolution block under inspection.
        for key in ("n_points", "time_step"):  # Entries required inside a resolution block.
            if key not in resolution:  # The entry is absent from the resolution block.
                raise ValueError(f"Solver resolution {name!r} is missing the key {key!r}")  # Reject.
    if solver["scheme"] not in ("imex_euler", "sbdf2"):  # Only two schemes are implemented.
        raise ValueError(f"Unknown time stepping scheme {solver['scheme']!r}")  # Reject clearly.
    for name in design["kernels"]:  # Every configured kernel family must be implemented.
        if name not in ("tophat", "gaussian"):  # Only two kernel families are implemented.
            raise ValueError(f"Unknown detection kernel {name!r} in the design block")  # Reject.


# Compute a stable hash of a configuration, used to label experiment output.
# Arguments:
#   config (dict): the configuration mapping to hash.
# Returns:
#   str: the first sixteen hexadecimal digits of the SHA-256 digest.
def config_hash(config):
    reduced = {key: value for key, value in config.items() if key != "config_path"}  # Drop the path.
    canonical = json.dumps(reduced, sort_keys=True, default=str)  # Canonical textual form.
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()  # Full hexadecimal digest.
    return digest[:16]  # A short prefix is sufficient to distinguish configurations.
