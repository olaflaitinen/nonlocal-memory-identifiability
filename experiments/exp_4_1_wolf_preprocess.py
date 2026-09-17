# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Preprocessing of the wolf telemetry of Section 6. The script
# verifies the Movebank files against the tracked checksum manifest, validates
# their schema, cleans the records, projects them to metric coordinates, runs
# the autocorrelated kernel density estimation through R, applies the inclusion
# rule and builds the study area of every included individual.
# Manuscript: Section 4.5 and Section 6; Table 7.
# Data: Latham ADM, Boutin S (2019), Movebank Data Repository,
# https://doi.org/10.5441/001/1.7vr1k987, licensed CC0 1.0.
# Inputs: the data package under data/raw and an experiment configuration.
# Outputs: the cleaned table, the study areas and a tracked summary.

import argparse  # Command line interface of the experiment script.
import subprocess  # Invocation of the R script of the density estimation.
import sys  # Process exit status of the experiment script.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and summary statistics.
import pandas  # Tabular handling of the Movebank attribute format.
from pyproj import Transformer  # Projection from geographic to metric coordinates.

from nmi.config import load_experiment_config  # Configuration loader with base inheritance.
from nmi.io import ensure_dir, write_csv  # Output writers of the experiment.
from nmi.wolf.schema import (  # Validation and cleaning of the Movebank data package.
    EVENT_FILE_NAMES,  # Accepted file names of the event file.
    REFERENCE_FILE_NAMES,  # Accepted file names of the reference file.
    clean_event_table,  # Cleaning rules of Section 4.5.
    load_event_table,  # Reader and validator of the event file.
    load_reference_table,  # Reader and validator of the reference file.
    locate_file,  # Locator of a file under several accepted names.
    pack_name,  # Pack name extracted from the free text of the reference file.
    verify_checksum,  # Verification against the tracked checksum manifest.
)
from nmi.wolf.study_area import build_study_area, buffer_width  # Study area construction.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Wolf preprocessing")  # Argument parser.
    parser.add_argument("--config", required=True, help="experiment configuration")  # Configuration.
    parser.add_argument("--output-dir", default=None, help="raw output directory")  # Destination.
    parser.add_argument("--skip-r", action="store_true", help="skip the density estimation")  # Skip.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# Verify the two files of the data package against the tracked manifest.
# Arguments:
#   event_path (pathlib.Path): the event file of the data package.
#   reference_path (pathlib.Path): the reference file of the data package.
#   settings (dict): the wolf block of the configuration.
# Returns:
#   dict: one entry per file stating whether its checksum matched.
def verify_files(event_path, reference_path, settings):
    manifest = settings["checksum_manifest"]  # Tracked manifest of the expected checksums.
    outcome = {  # Verification outcome of the two files of the data package.
        "event": verify_checksum(event_path, manifest),  # Checksum of the event file.
        "reference": verify_checksum(reference_path, manifest),  # Checksum of the reference file.
    }
    if settings.get("require_checksums", True) and not all(outcome.values()):  # A mismatch occurred.
        failed = [name for name, ok in outcome.items() if not ok]  # Files that failed the check.
        # A mismatch means that a different version of the package was downloaded.
        raise ValueError(f"Checksum verification failed for: {', '.join(failed)}; see data/README.md")
    return outcome  # Verification outcome, reported in the summary of the experiment.


# Regression checks against the known properties of the verified data package.
# Arguments:
#   events (pandas.DataFrame): the validated event table.
#   reference (pandas.DataFrame): the validated reference table.
# Returns:
#   dict: one entry per property, holding the observed value.
def regression_checks(events, reference):
    fix_type = pandas.to_numeric(events["gps:fix-type"], errors="coerce")  # Dimension of each fix.
    longitudes = pandas.to_numeric(events["location-long"], errors="coerce")  # Longitudes of fixes.
    # Sex of each animal, taken from the first deployment of that animal.
    sexes = reference.drop_duplicates(subset=["animal-id"])["animal-sex"].astype(str).str.lower()
    # Observed values of the properties recorded in data/README.md.
    return {
        "n_records": int(len(events)),  # Number of records in the event file.
        "n_animals": int(events["individual-local-identifier"].nunique()),  # Number of animals.
        "n_female": int((sexes == "f").sum()),  # Number of female animals in the reference file.
        "n_male": int((sexes == "m").sum()),  # Number of male animals in the reference file.
        "n_deployments": int(reference["deployment-id"].nunique()),  # Number of deployments.
        "all_visible": bool((events["visible"].astype(str).str.lower() == "true").all()),  # Flags.
        "n_two_dimensional_fixes": int((fix_type == 2).sum()),  # Two-dimensional GPS fixes.
        # Whether every fix lies within Universal Transverse Mercator zone 12.
        "longitudes_in_zone_12": bool(((longitudes >= -114.0) & (longitudes <= -108.0)).all()),
    }


# Entry point of the wolf preprocessing.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    config = load_experiment_config(arguments.config)  # Merged and validated configuration.
    if "wolf" not in config:  # The configuration does not describe the wolf application.
        # Report the mismatch rather than failing with an unclear key error.
        print("this configuration has no wolf block, use configs/exp_4_wolf.yaml")
        return 0  # Signal success, since an unsuitable configuration is not an error here.
    settings = config["wolf"]  # Settings of the application to the wolf data.
    raw = Path(settings["raw_directory"])  # Directory that holds the downloaded data package.
    event_path = locate_file(raw, EVENT_FILE_NAMES)  # Event file of the data package.
    reference_path = locate_file(raw, REFERENCE_FILE_NAMES)  # Reference file of the data package.
    checksums = verify_files(event_path, reference_path, settings)  # Verification of the files.
    events = load_event_table(event_path)  # Validated event table of the data package.
    reference = load_reference_table(reference_path)  # Validated reference table of the package.
    checks = regression_checks(events, reference)  # Regression checks of the known properties.
    cleaned, counts = clean_event_table(events)  # Cleaning rules of Section 4.5.
    transformer = Transformer.from_crs(  # Projection to the metric coordinate reference system.
        f"EPSG:{int(settings['source_epsg'])}",  # Geographic coordinates of the data package.
        f"EPSG:{int(settings['target_epsg'])}",  # Metric coordinates of the study area.
        always_xy=True,  # Interpret the coordinates as longitude and latitude in that order.
    )  # Transformer applied to every retained fix.
    easting, northing = transformer.transform(  # Projected coordinates of the retained fixes.
        cleaned["location-long"].to_numpy(dtype=float),  # Longitudes of the retained fixes.
        cleaned["location-lat"].to_numpy(dtype=float),  # Latitudes of the retained fixes.
    )  # Projected coordinates in metres.
    longitudes = cleaned["location-long"].to_numpy(dtype=float)  # Longitudes of the retained fixes.
    west = float(settings["utm_zone_west"])  # Western limit of the projection zone, in degrees.
    east = float(settings["utm_zone_east"])  # Eastern limit of the projection zone, in degrees.
    outside = float(np.mean((longitudes < west) | (longitudes > east)))  # Fraction outside the zone.
    print(f"fraction of fixes outside the configured projection zone: {outside:.4f}")  # Report it.
    table = pandas.DataFrame(  # Internal cleaned table written to the raw output directory.
        {  # Columns of the internal cleaned table of Section 4.5.
            "individual_id": cleaned["individual-local-identifier"].astype(str),  # Animal.
            "timestamp_utc": cleaned["timestamp"].astype(str),  # Time of the fix, in UTC.
            "easting_m": easting,  # Projected first coordinate of the fix, in metres.
            "northing_m": northing,  # Projected second coordinate of the fix, in metres.
        }
    )  # Cleaned and projected table of location fixes.
    directory = ensure_dir(arguments.output_dir or Path("results/raw/exp_4_1"))  # Output folder.
    cleaned_path = directory / "cleaned_fixes.csv"  # Internal cleaned table of Section 4.5.
    table.to_csv(cleaned_path, index=False)  # Write the internal cleaned table.
    print(f"wrote {cleaned_path} with {len(table)} retained fix(es)")  # Report the cleaned table.
    akde = run_density_estimation(settings, cleaned_path, directory, arguments.skip_r)  # AKDE step.
    # Summary table of the preprocessing, with one row per tracked animal.
    records = build_summary(table, reference, akde, settings, directory, checks, counts, checksums)
    summary = Path("results/summary") / f"{config['experiment']}_wolf_data.csv"  # Summary path.
    write_csv(summary, records, list(records[0].keys()))  # Write the tracked summary table.
    print(f"wrote {summary} with {len(records)} row(s)")  # Report the location of the summary.
    return 0  # Signal success to the caller.


# Run the autocorrelated kernel density estimation through the R script.
# Arguments:
#   settings (dict): the wolf block of the configuration.
#   cleaned_path (pathlib.Path): the internal cleaned table of location fixes.
#   directory (pathlib.Path): the directory that receives the estimator output.
#   skip (bool): whether the estimation is skipped on the command line.
# Returns:
#   dict: one entry per individual with the home range area and sample size.
def run_density_estimation(settings, cleaned_path, directory, skip):
    output = directory / "akde"  # Directory that receives the output of the R script.
    summary_path = output / "akde_summary.csv"  # Summary table written by the R script.
    if not skip and settings.get("run_r", True):  # The estimation was requested.
        command = [  # Command that runs the R script of the density estimation.
            str(settings.get("rscript_command", "Rscript")),  # Interpreter of the R script.
            str(settings.get("rscript_path", "R/akde_ctmm.R")),  # Path of the R script.
            str(cleaned_path),  # Internal cleaned table of location fixes.
            str(output),  # Directory that receives the estimator output.
            str(int(settings.get("akde_grid", 200))),  # Resolution of the exported raster.
        ]
        try:  # The R installation is optional and may be absent.
            subprocess.run(command, check=True)  # Run the R script of the density estimation.
        except (FileNotFoundError, subprocess.CalledProcessError) as error:  # R is unavailable.
            print(f"autocorrelated kernel density estimation was not run: {error}")  # Report it.
    if not summary_path.is_file():  # The estimation did not produce a summary table.
        print("no autocorrelated kernel density estimate is available")  # Report the absence.
        return {}  # The inclusion rule then reports the missing effective sample sizes.
    frame = pandas.read_csv(summary_path)  # Summary table written by the R script.
    estimates = {}  # Mapping from the animal identifier to its estimated quantities.
    for _, row in frame.iterrows():  # Record the estimate of each individual in turn.
        estimates[str(row["individual_id"])] = {  # Estimated quantities of this individual.
            "area_m2": float(row["area_m2"]),  # Ninety five per cent home range area.
            "n_eff": float(row["n_eff"]),  # Effective sample size for area.
            "ud_path": str(row.get("ud_path", "")),  # Path of the utilisation distribution.
        }
    return estimates  # Estimated quantities of every individual.


# Build the summary table of the preprocessing, one row per individual.
# Arguments:
#   table (pandas.DataFrame): the internal cleaned table of location fixes.
#   reference (pandas.DataFrame): the validated reference table of deployments.
#   akde (dict): the estimated home range areas and effective sample sizes.
#   settings (dict): the wolf block of the configuration.
#   directory (pathlib.Path): the directory that receives the study areas.
#   checks (dict): the regression checks of the known data properties.
#   counts (dict): the number of records removed by each cleaning rule.
#   checksums (dict): the verification outcome of the two files.
# Returns:
#   list: one summary record per individual, in ascending identifier order.
def build_summary(table, reference, akde, settings, directory, checks, counts, checksums):
    reference_index = {}  # Mapping from the animal identifier to its pack and sex.
    for _, row in reference.iterrows():  # Index the reference table by animal identifier.
        identifier = str(row["animal-id"])  # Identifier of the animal of this deployment.
        reference_index.setdefault(  # Keep the first deployment of each animal.
            identifier,  # Identifier under which the entry is stored.
            {  # Pack and sex of the animal, taken from the reference file.
                "pack": pack_name(row.get("animal-comments")),  # Pack name before the semicolon.
                "sex": str(row.get("animal-sex", "")),  # Sex of the animal.
            },
        )  # Entry of this animal in the index.
    records = []  # Accumulator for the summary rows of the individuals.
    for identifier, group in table.groupby("individual_id"):  # Summarise each individual.
        times = pandas.to_datetime(group["timestamp_utc"], utc=True)  # Times of the retained fixes.
        span_days = float((times.max() - times.min()).total_seconds() / 86400.0)  # Monitoring span.
        # Intervals between consecutive fixes of this animal, in seconds.
        intervals = np.diff(np.sort(times.to_numpy().astype("datetime64[s]").astype(float)))
        # Median interval between consecutive fixes, reported in hours.
        median_interval = float(np.median(intervals) / 3600.0) if intervals.size else float("nan")
        estimate = akde.get(str(identifier), {})  # Estimated quantities of this individual.
        n_eff = float(estimate.get("n_eff", float("nan")))  # Effective sample size for area.
        area = float(estimate.get("area_m2", float("nan")))  # Ninety five per cent home range area.
        long_enough = span_days >= float(settings["min_span_days"])  # Monitoring span criterion.
        # Whether the effective sample size satisfies the inclusion rule.
        informative = np.isfinite(n_eff) and n_eff >= float(settings["min_effective_size"])
        included = bool(long_enough and informative)  # Inclusion rule fixed before any fitting.
        reason = ""  # Reason for exclusion, empty for an included individual.
        if not long_enough:  # The monitoring span is shorter than the configured minimum.
            # Reason for excluding an individual with a short monitoring span.
            reason = f"monitoring span of {span_days:.1f} days is shorter than the minimum"
        elif not informative:  # The effective sample size is below the configured minimum.
            # Reason for excluding an individual with uninformative fixes.
            reason = "effective sample size for area is below the minimum or is unavailable"
        study_area_cells = ""  # Number of interior cells of the study area, when it is built.
        if included and np.isfinite(area):  # A study area is built for the included individuals.
            width = buffer_width(area, float(settings["buffer_factor"]))  # Buffer of Section 4.5.
            area_map = build_study_area(  # Masked Cartesian grid of the study area.
                group["easting_m"].to_numpy(dtype=float),  # Projected first coordinate.
                group["northing_m"].to_numpy(dtype=float),  # Projected second coordinate.
                width,  # Width of the buffer around the enclosing region.
                float(settings["cell_metres"]),  # Cell width of the Cartesian grid.
            )  # Study area of this individual.
            study_area_cells = int(np.sum(area_map["mask"]))  # Interior cells of the study area.
            np.savez(  # Store the study area for the fitting scripts of Experiments 4.2 and 4.3.
                directory / f"study_area_{identifier}.npz",  # Archive of this individual.
                mask=area_map["mask"],  # Boolean mask of the study area.
                x_centres=area_map["x_centres"],  # Cell centres along the first axis.
                y_centres=area_map["y_centres"],  # Cell centres along the second axis.
                cell=np.array([area_map["cell"]]),  # Cell width of the Cartesian grid.
            )  # Archive written for the later experiments.
        info = reference_index.get(str(identifier), {"pack": "unknown", "sex": ""})  # Metadata.
        records.append(  # One summary row of this individual.
            {  # Summary of the telemetry and of the inclusion decision.
                "individual_id": str(identifier),  # Identifier of the animal.
                "pack": info["pack"],  # Pack name taken from the reference file.
                "sex": info["sex"],  # Sex of the animal taken from the reference file.
                "first_fix": str(times.min()),  # Time of the first retained fix.
                "last_fix": str(times.max()),  # Time of the last retained fix.
                "n_fixes": int(len(group)),  # Number of retained fixes of the animal.
                "median_interval_hours": median_interval,  # Median interval between two fixes.
                "span_days": span_days,  # Monitoring span from the first to the last fix.
                "n_eff": n_eff,  # Effective sample size for area from the density estimation.
                "home_range_km2": area / 1.0e6 if np.isfinite(area) else "",  # Home range area.
                "study_area_cells": study_area_cells,  # Interior cells of the study area.
                "included": included,  # Whether the individual satisfies the inclusion rule.
                "exclusion_reason": reason,  # Reason for exclusion, empty when included.
                "checksum_event_ok": checksums["event"],  # Verification of the event file.
                "checksum_reference_ok": checksums["reference"],  # Verification of the reference.
                "check_n_records": checks["n_records"],  # Observed number of records.
                "check_n_animals": checks["n_animals"],  # Observed number of animals.
                "check_n_female": checks["n_female"],  # Observed number of female animals.
                "check_n_male": checks["n_male"],  # Observed number of male animals.
                "check_n_deployments": checks["n_deployments"],  # Observed number of deployments.
                "check_all_visible": checks["all_visible"],  # Whether every record is visible.
                "check_two_dimensional_fixes": checks["n_two_dimensional_fixes"],  # Fix types.
                "check_longitudes_in_zone_12": checks["longitudes_in_zone_12"],  # Zone check.
                "dropped_duplicate_timestamps": counts["dropped_duplicate_timestamps"],  # Rule.
                "dropped_not_visible": counts["dropped_not_visible"],  # Cleaning rule count.
                "dropped_non_gps": counts["dropped_non_gps"],  # Cleaning rule count.
                # Citation of the data package, carried through every output.
                "data_citation": "Latham and Boutin (2019) https://doi.org/10.5441/001/1.7vr1k987",
            }
        )  # Row appended to the summary table.
    records.sort(key=lambda record: record["individual_id"])  # Stable order of the summary rows.
    return records  # Summary table of the preprocessing of Experiment 4.1.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Run the experiment and propagate the exit status.
