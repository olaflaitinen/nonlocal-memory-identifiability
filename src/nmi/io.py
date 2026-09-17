# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Writers for the outputs of the experiments. Every output carries the
# metadata required for reproduction: the git commit of the working tree, the
# hash of the configuration, the versions of the scientific libraries, the seed
# of the condition and the time of the run.
# Manuscript: Section 4.6 and docs/reproducibility.md.
# Inputs: arrays, tables and configurations. Outputs: HDF5 archives and comma
# separated value files under results/.

import csv  # Reader and writer of the comma separated summary files.
import datetime  # Coordinated universal time stamps of the runs.
import importlib.metadata  # Versions of the installed scientific libraries.
import subprocess  # Reading of the git commit of the working tree.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays written into the archives.

from nmi.config import config_hash  # Stable hash of a configuration mapping.

# Libraries whose versions are recorded in the metadata of every output.
RECORDED_LIBRARIES = ("numpy", "scipy", "emcee", "arviz", "h5py", "pandas")  # Scientific stack.


# Current git commit of the working tree, or a placeholder when unavailable.
# Arguments:
#   none.
# Returns:
#   str: the forty character commit hash, or "unknown" outside a repository.
def git_commit():
    command = ["git", "rev-parse", "HEAD"]  # Command that prints the current commit.
    try:  # The code may run from an archive that carries no git metadata.
        output = subprocess.run(command, capture_output=True, text=True, check=True)  # Run git.
    except (subprocess.CalledProcessError, FileNotFoundError):  # No repository or no git binary.
        return "unknown"  # Record the absence of version control information.
    return output.stdout.strip()  # The commit hash of the working tree.


# Metadata recorded alongside every experiment output.
# Arguments:
#   config (dict): the configuration mapping of the run.
#   extra (dict): further entries to record, for example the condition seed.
# Returns:
#   dict: a flat mapping of textual metadata entries.
def run_metadata(config, extra=None):
    versions = {}  # Versions of the recorded scientific libraries.
    for name in RECORDED_LIBRARIES:  # Record one entry per library of the stack.
        try:  # A library may be absent from a minimal environment.
            versions[name] = importlib.metadata.version(name)  # Installed version string.
        except importlib.metadata.PackageNotFoundError:  # The library is not installed.
            versions[name] = "absent"  # Record the absence rather than failing the run.
    stamp = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")  # Time of the run.
    # Core metadata entries shared by every output of the repository.
    metadata = {
        "git_commit": git_commit(),  # Commit of the working tree that produced the output.
        "config_hash": config_hash(config),  # Stable hash of the configuration mapping.
        "config_path": str(config.get("config_path", "unknown")),  # Origin of the configuration.
        "experiment": str(config.get("experiment", "unknown")),  # Identifier of the experiment.
        "utc_time": stamp,  # Coordinated universal time at which the output was written.
    }
    for name, version in versions.items():  # Record the version of each scientific library.
        metadata[f"version_{name}"] = version  # One metadata entry per library.
    if extra:  # Further entries supplied by the calling experiment script.
        for key, value in extra.items():  # Record each additional entry as text.
            metadata[str(key)] = str(value)  # Metadata values are stored as strings.
    return metadata  # Flat mapping written into every output file.


# Create a directory, including any missing parent directories.
# Arguments:
#   path (str or pathlib.Path): the directory to create.
# Returns:
#   pathlib.Path: the created directory.
def ensure_dir(path):
    location = Path(path)  # Normalise the argument into a path object.
    location.mkdir(parents=True, exist_ok=True)  # Create the directory when it is absent.
    return location  # Return the directory for convenience.


# Write a table of dictionaries to a comma separated value file.
# Arguments:
#   path (str or pathlib.Path): the file to write.
#   rows (sequence): dictionaries with identical keys.
#   fieldnames (sequence): the column names, in the order of the output.
# Returns:
#   pathlib.Path: the file that was written.
def write_csv(path, rows, fieldnames=None):
    location = Path(path)  # Normalise the argument into a path object.
    ensure_dir(location.parent)  # Create the containing directory when it is absent.
    columns = list(fieldnames) if fieldnames else list(rows[0].keys())  # Column names of the table.
    with location.open("w", encoding="utf-8", newline="") as handle:  # Open the output file.
        writer = csv.DictWriter(handle, fieldnames=columns)  # Writer bound to the column names.
        writer.writeheader()  # Write the header row of the table.
        for row in rows:  # Write one line per record of the table.
            writer.writerow({key: row.get(key, "") for key in columns})  # Selected columns only.
    return location  # Return the written file for convenience.


# Write arrays and metadata to an HDF5 archive.
# Arguments:
#   path (str or pathlib.Path): the archive to write.
#   arrays (dict): named arrays stored as datasets.
#   metadata (dict): textual metadata stored as attributes of the root group.
# Returns:
#   pathlib.Path: the archive that was written.
def write_hdf5(path, arrays, metadata):
    import h5py  # Imported lazily so that the package imports without HDF5 support.

    location = Path(path)  # Normalise the argument into a path object.
    ensure_dir(location.parent)  # Create the containing directory when it is absent.
    with h5py.File(location, "w") as handle:  # Open the archive for writing.
        for name, value in arrays.items():  # Store one dataset per named array.
            handle.create_dataset(name, data=np.asarray(value))  # Write the array itself.
        for key, value in metadata.items():  # Store the metadata as root attributes.
            handle.attrs[str(key)] = str(value)  # Attributes are stored as strings.
    return location  # Return the written archive for convenience.


# Read arrays and metadata from an HDF5 archive.
# Arguments:
#   path (str or pathlib.Path): the archive to read.
# Returns:
#   tuple: a mapping of named arrays and a mapping of textual metadata.
def read_hdf5(path):
    import h5py  # Imported lazily so that the package imports without HDF5 support.

    location = Path(path)  # Normalise the argument into a path object.
    arrays = {}  # Accumulator for the datasets of the archive.
    metadata = {}  # Accumulator for the attributes of the root group.
    with h5py.File(location, "r") as handle:  # Open the archive for reading.
        for name in handle:  # Read every dataset stored at the root of the archive.
            arrays[name] = np.array(handle[name])  # Materialise the dataset as an array.
        for key, value in handle.attrs.items():  # Read every attribute of the root group.
            metadata[key] = str(value)  # Attributes are stored and returned as strings.
    return arrays, metadata  # Datasets and metadata of the archive.


# Write the metadata of a run into a small comma separated file.
# Arguments:
#   path (str or pathlib.Path): the file to write.
#   metadata (dict): the mapping produced by run_metadata.
# Returns:
#   pathlib.Path: the file that was written.
def write_metadata_csv(path, metadata):
    rows = [{"key": key, "value": value} for key, value in sorted(metadata.items())]  # Long form.
    return write_csv(path, rows, ["key", "value"])  # Two column table of metadata entries.


# Read the generated synthetic data of one condition from its archive.
# Arguments:
#   experiment (str): the experiment identifier under which the data were written.
#   index (int): the stable integer index of the condition.
#   directory (str or pathlib.Path): the directory that holds the archives.
# Returns:
#   dict: the transient and stationary observations, the noise scale and the
#   largest density of the reference simulation.
def read_condition_data(experiment, index, directory=None):
    folder = Path(directory) if directory else Path("results/raw") / str(experiment)  # Location.
    arrays, metadata = read_hdf5(folder / f"data_{int(index):03d}.h5")  # Contents of the archive.
    # Observations and noise scale of the requested condition.
    return {
        "observations": arrays["observations"],  # Noisy transient observations.
        "clean": arrays["clean"],  # Noise free transient values at the same points.
        "times": arrays["times"],  # Observation times of the sampling design.
        "stationary_observations": arrays["stationary_observations"],  # Noisy stationary values.
        "stationary_clean": arrays["stationary_clean"],  # Noise free stationary values.
        "stationary_density": arrays["stationary_density"],  # Stationary density on the grid.
        "sigma": float(metadata["sigma"]),  # Standard deviation of the observation noise.
        "u_max": float(metadata["u_max"]),  # Largest density of the reference simulation.
        "seed": int(metadata["seed"]),  # Deterministic seed of the condition.
    }


# Write a table as a LaTeX tabular environment in the booktabs style.
# The output is compatible with the Springer Nature LaTeX template.
# Arguments:
#   path (str or pathlib.Path): the file to write.
#   rows (sequence): dictionaries with identical keys.
#   columns (sequence): the column names, in the order of the output.
#   headers (sequence): the column headers shown in the typeset table.
#   caption (str): the caption of the table.
#   label (str): the LaTeX label of the table.
# Returns:
#   pathlib.Path: the file that was written.
def write_latex(path, rows, columns, headers, caption, label):
    location = Path(path)  # Normalise the argument into a path object.
    ensure_dir(location.parent)  # Create the containing directory when it is absent.
    lines = ["\\begin{table}[t]", "\\centering"]  # Opening of the table environment.
    lines.append(f"\\caption{{{caption}}}")  # Caption of the typeset table.
    lines.append(f"\\label{{{label}}}")  # Label used for cross references.
    lines.append("\\begin{tabular}{" + "l" * len(columns) + "}")  # Column specification.
    lines.append("\\toprule")  # Upper rule of the booktabs style.
    lines.append(" & ".join(str(item) for item in headers) + " \\\\")  # Header row.
    lines.append("\\midrule")  # Middle rule of the booktabs style.
    for row in rows:  # Write one line per record of the table.
        cells = [str(row.get(name, "")) for name in columns]  # Cells of the current row.
        lines.append(" & ".join(cells) + " \\\\")  # Body row of the typeset table.
    lines.append("\\bottomrule")  # Lower rule of the booktabs style.
    lines.append("\\end{tabular}")  # End of the tabular environment.
    lines.append("\\end{table}")  # End of the table environment.
    location.write_text("\n".join(lines) + "\n", encoding="utf-8")  # Write the LaTeX source.
    return location  # Return the written file for convenience.


# Read a tracked summary table into a list of dictionaries.
# Arguments:
#   path (str or pathlib.Path): the comma separated file to read.
# Returns:
#   list: one dictionary per row, or an empty list when the file is absent.
def read_summary(path):
    location = Path(path)  # Normalise the argument into a path object.
    if not location.is_file():  # The summary has not been produced yet.
        return []  # Report the absence as an empty table.
    with location.open(encoding="utf-8", newline="") as handle:  # Open the summary for reading.
        return list(csv.DictReader(handle))  # One dictionary per row of the table.


# Read the summary table of the wolf preprocessing of Experiment 4.1.
# Arguments:
#   experiment (str): the experiment identifier used in the file name.
#   folder (str or pathlib.Path): the directory that holds the summaries.
# Returns:
#   list: one dictionary per individual, or an empty list when it is absent.
def read_wolf_summary(experiment, folder="results/summary"):
    return read_summary(Path(folder) / f"{experiment}_wolf_data.csv")  # Summary of Experiment 4.1.
