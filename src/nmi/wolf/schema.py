# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Validation of the Movebank data package used in Section 6. The
# module records the required columns of the event file and of the reference
# file, locates the files under data/raw, verifies their checksums and cleans
# the records as specified in Section 4.5.
# Manuscript: Section 4.5, inference for the wolf data; Section 6, Table 7.
# Data: Latham ADM, Boutin S (2019), Movebank Data Repository,
# https://doi.org/10.5441/001/1.7vr1k987, licensed CC0 1.0.
# Inputs: the two files of the data package. Outputs: validated and cleaned
# tables of location fixes.

import hashlib  # Verification of the checksums of the downloaded files.
from pathlib import Path  # Portable filesystem paths.

import pandas  # Tabular handling of the Movebank attribute format.

# Columns that the Movebank event file must provide.
EVENT_COLUMNS = (
    "event-id",  # Unique identifier of one recorded event.
    "visible",  # Flag that marks records hidden by the data owner.
    "timestamp",  # Time of the fix, recorded in coordinated universal time.
    "location-long",  # Longitude of the fix in the World Geodetic System 1984.
    "location-lat",  # Latitude of the fix in the World Geodetic System 1984.
    "gps:dop",  # Dilution of precision reported by the collar.
    "gps:fix-type",  # Dimension of the fix, two or three.
    "gps:satellite-count",  # Number of satellites used for the fix.
    "sensor-type",  # Sensor that produced the record.
    "individual-taxon-canonical-name",  # Scientific name of the tracked species.
    "tag-local-identifier",  # Identifier of the collar within the study.
    "individual-local-identifier",  # Identifier of the animal within the study.
    "study-name",  # Name of the study in the Movebank repository.
)  # End of the required event columns.

# Columns that the Movebank reference file must provide.
REFERENCE_COLUMNS = (
    "animal-id",  # Identifier of the animal, equal to individual-local-identifier.
    "deployment-id",  # Identifier of one deployment of a collar on an animal.
    "animal-sex",  # Sex of the animal, recorded as m or f.
    "animal-life-stage",  # Life stage of the animal at collaring.
    "animal-comments",  # Free text whose first field is the pack name.
    "deploy-on-date",  # Date on which the collar was deployed.
    "deploy-off-date",  # Date on which the collar was removed or stopped.
    "deployment-end-type",  # Reason for the end of the deployment.
    "duty-cycle",  # Programmed fix interval of the collar.
)  # End of the required reference columns.

# Accepted file names of the event file of the data package.
EVENT_FILE_NAMES = (
    "Latham Alberta Wolves.csv",  # Original name as published in the repository.
    "Latham_Alberta_Wolves.csv",  # Same file with underscores instead of spaces.
)  # End of the accepted event file names.

# Accepted file names of the reference file of the data package.
REFERENCE_FILE_NAMES = (
    "Latham Alberta Wolves-reference-data.csv",  # Original name as published.
    "Latham_Alberta_Wolves-reference-data.csv",  # Same file with underscores.
)  # End of the accepted reference file names.


# Locate one of several accepted file names inside a directory.
# Arguments:
#   directory (str or pathlib.Path): the directory that holds the data package.
#   names (sequence): the accepted file names, in order of preference.
# Returns:
#   pathlib.Path: the first accepted file that exists in the directory.
def locate_file(directory, names):
    folder = Path(directory)  # Normalise the argument into a path object.
    for name in names:  # Inspect the accepted file names in order of preference.
        candidate = folder / name  # Candidate path inside the data directory.
        if candidate.is_file():  # The file of the data package is present.
            return candidate  # Use the first accepted file that exists.
    accepted = " or ".join(repr(name) for name in names)  # Human readable list of the names.
    # The data package is not committed, so an informative message is essential.
    raise FileNotFoundError(f"Expected {accepted} under {folder}, see data/README.md for the download")


# Compute the SHA-256 checksum of a file.
# Arguments:
#   path (str or pathlib.Path): the file whose checksum is required.
# Returns:
#   str: the hexadecimal digest of the file contents.
def file_checksum(path):
    digest = hashlib.sha256()  # Incremental hash of the file contents.
    with Path(path).open("rb") as handle:  # Open the file in binary mode.
        for block in iter(lambda: handle.read(1 << 20), b""):  # Read the file in one megabyte blocks.
            digest.update(block)  # Feed the block into the incremental hash.
    return digest.hexdigest()  # Hexadecimal digest of the whole file.


# Verify the checksum of a file against the tracked manifest.
# Arguments:
#   path (str or pathlib.Path): the file to verify.
#   manifest (str or pathlib.Path): the manifest data/checksums.sha256.
# Returns:
#   bool: True when the manifest lists the file and the checksum matches.
def verify_checksum(path, manifest="data/checksums.sha256"):
    expected = {}  # Mapping from file name to the expected checksum.
    manifest_path = Path(manifest)  # Normalise the argument into a path object.
    if not manifest_path.is_file():  # The manifest has not been created yet.
        return False  # Without a manifest no verification is possible.
    for line in manifest_path.read_text(encoding="utf-8").splitlines():  # Read the manifest.
        parts = line.split()  # A manifest line holds a checksum and a file name.
        if len(parts) >= 2:  # Ignore blank lines and malformed entries.
            expected[parts[-1].lstrip("*")] = parts[0]  # Record the expected checksum.
    candidates = (Path(path).name, Path(path).name.replace(" ", "_"))  # Both accepted spellings.
    for name in candidates:  # Look the file up under either accepted spelling.
        if name in expected:  # The manifest lists this file.
            return file_checksum(path) == expected[name]  # Compare the two checksums.
    return False  # The manifest does not list the file at all.


# Validate that a table provides every required column.
# Arguments:
#   frame (pandas.DataFrame): the table to validate.
#   columns (sequence): the required column names.
#   label (str): a human readable name of the table, used in the message.
# Returns:
#   None: the function returns quietly or raises ValueError.
def validate_columns(frame, columns, label):
    missing = [name for name in columns if name not in frame.columns]  # Absent column names.
    if missing:  # At least one required column is absent from the table.
        listed = ", ".join(missing)  # Human readable list of the absent columns.
        raise ValueError(f"The {label} is missing the required column(s): {listed}")  # Reject.


# Read and validate the Movebank event file.
# Arguments:
#   path (str or pathlib.Path): the event file of the data package.
# Returns:
#   pandas.DataFrame: the validated table, with the timestamp parsed as UTC.
def load_event_table(path):
    frame = pandas.read_csv(path, low_memory=False)  # Read the Movebank attribute format.
    validate_columns(frame, EVENT_COLUMNS, "Movebank event file")  # Schema of Section 4.5.
    frame["timestamp"] = pandas.to_datetime(frame["timestamp"], utc=True)  # Parse as UTC.
    return frame  # Validated event table of the data package.


# Read and validate the Movebank reference file.
# Arguments:
#   path (str or pathlib.Path): the reference file of the data package.
# Returns:
#   pandas.DataFrame: the validated table of deployments.
def load_reference_table(path):
    frame = pandas.read_csv(path, low_memory=False)  # Read the Movebank attribute format.
    validate_columns(frame, REFERENCE_COLUMNS, "Movebank reference file")  # Schema of Section 4.5.
    return frame  # Validated reference table of the data package.


# Pack name of an animal, taken from the free text of the reference file.
# Arguments:
#   comment (str): the value of the column animal-comments.
# Returns:
#   str: the text before the first semicolon, or "unknown" when absent.
def pack_name(comment):
    if not isinstance(comment, str) or not comment.strip():  # No comment was recorded.
        return "unknown"  # Report the absence rather than inventing a pack name.
    return comment.split(";")[0].strip()  # Pack name precedes the first semicolon.


# Clean the event table as specified in Section 4.5 of the manuscript.
# Arguments:
#   frame (pandas.DataFrame): the validated event table.
# Returns:
#   tuple: the cleaned table and a mapping of the counts removed by each rule.
def clean_event_table(frame):
    counts = {"initial": int(len(frame))}  # Number of records before any rule is applied.
    cleaned = frame[frame["sensor-type"].astype(str).str.lower() == "gps"].copy()  # Keep GPS only.
    counts["dropped_non_gps"] = counts["initial"] - int(len(cleaned))  # Records of other sensors.
    before = int(len(cleaned))  # Number of records before the visibility rule is applied.
    cleaned = cleaned[cleaned["visible"].astype(str).str.lower() == "true"].copy()  # Visible only.
    counts["dropped_not_visible"] = before - int(len(cleaned))  # Records flagged as not visible.
    before = int(len(cleaned))  # Number of records before the identifier rule is applied.
    cleaned = cleaned[cleaned["individual-local-identifier"].notna()].copy()  # Identified animals.
    counts["dropped_unidentified"] = before - int(len(cleaned))  # Records without an animal.
    before = int(len(cleaned))  # Number of records before the coordinate rule is applied.
    # Records without coordinates carry no spatial information and are removed.
    cleaned = cleaned[cleaned["location-long"].notna() & cleaned["location-lat"].notna()].copy()
    counts["dropped_without_location"] = before - int(len(cleaned))  # Records without coordinates.
    before = int(len(cleaned))  # Number of records before the duplicate rule is applied.
    keys = ["individual-local-identifier", "timestamp"]  # Duplicates are defined per animal.
    cleaned = cleaned.sort_values(keys).drop_duplicates(subset=keys, keep="first").copy()  # Keep first.
    counts["dropped_duplicate_timestamps"] = before - int(len(cleaned))  # Duplicate timestamps.
    counts["retained"] = int(len(cleaned))  # Number of records retained for the analysis.
    return cleaned.reset_index(drop=True), counts  # Cleaned table and the counts of each rule.
