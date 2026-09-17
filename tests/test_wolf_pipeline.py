# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Verify the wolf pipeline without using the data package. The schema
# validation is exercised on a synthetic table in the Movebank attribute format
# and the masked likelihood is exercised on a synthetic point pattern on an
# artificial domain.
# Manuscript: Section 4.5; Section 6.
# Inputs: none, the tests generate their own synthetic data in code.
# Outputs: assertions of the test runner.

import numpy as np  # Numerical arrays and random number generation.
import pandas  # Tabular handling of the synthetic Movebank tables.
import pytest  # Approximate comparison and expected error declaration.

from nmi.wolf.masked_solver import convolve_masked, masked_steady_state  # Masked domain solver.
from nmi.wolf.model_comparison import block_indices, compare_models, ranking  # Model comparison.
from nmi.wolf.point_likelihood import (  # Weighted point likelihood of equation (10).
    pointwise_log_density,  # Log density of the fitted model at every fix.
    uniform_loglik,  # Weighted log-likelihood of the model without nonlocal advection.
    weighted_point_loglik,  # Weighted log-likelihood of equation (10).
)
from nmi.wolf.schema import (  # Validation and cleaning of the Movebank data package.
    EVENT_COLUMNS,  # Columns required by the Movebank event file.
    clean_event_table,  # Cleaning rules of Section 4.5.
    pack_name,  # Pack name extracted from the free text of the reference file.
    validate_columns,  # Schema validation of a Movebank table.
)
from nmi.wolf.study_area import build_study_area, buffer_width, cell_indices  # Study area.


# Build a synthetic table in the Movebank attribute format.
# Arguments:
#   n_records (int): the number of records of the synthetic table.
# Returns:
#   pandas.DataFrame: a table that carries every required column.
def synthetic_event_table(n_records=40):
    rng = np.random.default_rng(5)  # Deterministic generator of the synthetic records.
    times = pandas.date_range("2006-02-01", periods=n_records, freq="2h", tz="UTC")  # Timestamps.
    frame = pandas.DataFrame(  # Synthetic table in the Movebank attribute format.
        {  # One column per attribute required by the schema of Section 4.5.
            "event-id": np.arange(n_records),  # Unique identifier of each record.
            "visible": ["true"] * n_records,  # Every synthetic record is visible.
            "timestamp": times,  # Time of each synthetic fix.
            "location-long": -112.0 + 0.01 * rng.standard_normal(n_records),  # Longitudes.
            "location-lat": 56.0 + 0.01 * rng.standard_normal(n_records),  # Latitudes.
            "gps:dop": rng.uniform(1.0, 5.0, n_records),  # Dilution of precision of each fix.
            "gps:fix-type": rng.integers(2, 4, n_records),  # Dimension of each synthetic fix.
            "gps:satellite-count": rng.integers(4, 10, n_records),  # Satellites used per fix.
            "sensor-type": ["gps"] * n_records,  # Every synthetic record is a GPS record.
            "individual-taxon-canonical-name": ["Canis lupus"] * n_records,  # Species name.
            "tag-local-identifier": ["tag-1"] * n_records,  # Identifier of the synthetic collar.
            "individual-local-identifier": ["1"] * n_records,  # Identifier of the animal.
            "study-name": ["synthetic"] * n_records,  # Name of the synthetic study.
        }
    )  # Synthetic event table used by the schema tests.
    return frame  # Synthetic table that carries every required column.


# A table that carries every required column passes the schema validation.
# Arguments:
#   none.
# Returns:
#   None: the test asserts that no error is raised.
def test_schema_accepts_a_complete_table():
    frame = synthetic_event_table()  # Synthetic table in the Movebank attribute format.
    validate_columns(frame, EVENT_COLUMNS, "synthetic event file")  # Validation of the schema.


# A table that lacks a required column is rejected with an informative message.
# Arguments:
#   none.
# Returns:
#   None: the test asserts that the error names the missing column.
def test_schema_rejects_a_missing_column():
    frame = synthetic_event_table().drop(columns=["gps:fix-type"])  # Remove a required column.
    with pytest.raises(ValueError, match="gps:fix-type"):  # The error names the missing column.
        validate_columns(frame, EVENT_COLUMNS, "synthetic event file")  # Validation of the schema.


# The cleaning rules of Section 4.5 remove the documented records.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected counts.
def test_cleaning_rules():
    frame = synthetic_event_table()  # Synthetic table in the Movebank attribute format.
    frame.loc[0, "visible"] = "false"  # One record is flagged as not visible.
    frame.loc[1, "sensor-type"] = "argos-doppler-shift"  # One record is not a GPS record.
    duplicated = pandas.concat([frame, frame.iloc[[5]]], ignore_index=True)  # One duplicate time.
    cleaned, counts = clean_event_table(duplicated)  # Cleaning rules of Section 4.5.
    assert counts["dropped_not_visible"] == 1  # Exactly one record was flagged as not visible.
    assert counts["dropped_non_gps"] == 1  # Exactly one record came from another sensor.
    assert counts["dropped_duplicate_timestamps"] == 1  # Exactly one duplicate timestamp remained.
    assert counts["retained"] == len(cleaned)  # The reported count matches the cleaned table.


# The pack name is taken from the free text of the reference file.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected pack names.
def test_pack_name_extraction():
    assert pack_name("GoCan; collared in 2006") == "GoCan"  # Text before the first semicolon.
    assert pack_name("Loner") == "Loner"  # The dispersing wolf carries no further text.
    assert pack_name(None) == "unknown"  # A missing comment is reported as unknown.


# The study area encloses every fix and carries the requested buffer.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected geometry.
def test_study_area_contains_every_fix():
    rng = np.random.default_rng(8)  # Deterministic generator of the synthetic point pattern.
    easting = 400000.0 + 4000.0 * rng.standard_normal(200)  # Projected first coordinate.
    northing = 6200000.0 + 4000.0 * rng.standard_normal(200)  # Projected second coordinate.
    width = buffer_width(1.0e8, 0.5)  # Buffer width implied by a home range of one hundred km2.
    assert width == pytest.approx(5000.0)  # Half of the square root of the home range area.
    area = build_study_area(easting, northing, width, 500.0)  # Masked grid of the study area.
    rows, columns = cell_indices(easting, northing, area)  # Cells that contain the fixes.
    assert bool(np.all(area["mask"][rows, columns]))  # Every fix lies inside the study area.
    assert area["area"] > 0.0  # The discretised study area has a positive area.


# The masked likelihood prefers the model that generated the synthetic points.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected ordering of the log-likelihoods.
def test_masked_likelihood_prefers_the_generating_model():
    mask = np.ones((40, 40), dtype=bool)  # Artificial square domain of the test.
    cell = 100.0  # Cell width of the artificial domain, in metres.
    truth = masked_steady_state(4.0, 600.0, "tophat", mask, cell)  # Generating stationary density.
    assert truth["converged"]  # The fixed-point solver reached the requested tolerance.
    rng = np.random.default_rng(21)  # Deterministic generator of the synthetic point pattern.
    flat = truth["density"].ravel() / float(np.sum(truth["density"]))  # Probability of each cell.
    picks = rng.choice(flat.size, size=600, p=flat)  # Cells that contain the synthetic points.
    rows, columns = np.unravel_index(picks, truth["density"].shape)  # Indices of those cells.
    area = {  # Study area mapping in the layout used by the point likelihood.
        "mask": mask,  # Boolean mask of the artificial domain.
        "x_centres": (np.arange(mask.shape[1]) + 0.5) * cell,  # Cell centres, first axis.
        "y_centres": (np.arange(mask.shape[0]) + 0.5) * cell,  # Cell centres, second axis.
        "cell": cell,  # Cell width of the artificial domain.
    }  # Study area of the artificial domain.
    easting = area["x_centres"][columns]  # First coordinate of the synthetic points.
    northing = area["y_centres"][rows]  # Second coordinate of the synthetic points.
    individual = {  # Data of the artificial individual of the test.
        "easting": easting,  # First coordinate of the synthetic points.
        "northing": northing,  # Second coordinate of the synthetic points.
        "n_eff": 60.0,  # Effective sample size of the artificial individual.
        "kernel": "tophat",  # Detection kernel family of the generating model.
        "area": area,  # Study area of the artificial domain.
    }  # Artificial individual used by the comparison.
    fitted = pointwise_log_density(easting, northing, truth["density"], area)  # Generating model.
    generating = weighted_point_loglik(fitted, individual["n_eff"])  # Weighted log-likelihood.
    uniform, _ = uniform_loglik(individual)  # Weighted log-likelihood of the uniform model.
    assert generating > uniform  # The generating model fits the synthetic points better.


# The masked convolution agrees with a direct sum over the cells.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected agreement.
def test_masked_convolution_matches_direct_sum():
    from nmi.kernels import kernel_values_2d  # Profile of the two-dimensional detection kernel.

    size, cell, radius = 10, 50.0, 120.0  # Small grid used by the comparison.
    field = np.random.default_rng(13).random((size, size))  # Arbitrary field on the small grid.
    computed = convolve_masked(field, radius, "gaussian", cell)  # Convolution under test.
    direct = np.zeros((size, size))  # Accumulator for the direct double sum.
    axis = np.arange(size)  # Indices of the cells along each coordinate axis.
    for row in range(size):  # Evaluate the convolution at each cell of the grid.
        for column in range(size):  # Evaluate the convolution at each column of that row.
            # Distances from the current cell centre to every cell of the grid.
            distance = np.hypot((axis[:, None] - row) * cell, (axis[None, :] - column) * cell)
            weights = kernel_values_2d(distance, radius, "gaussian")  # Quadrature weights.
            direct[row, column] = float(np.sum(weights * field) * cell * cell)  # Direct sum.
    assert np.allclose(computed, direct, atol=1.0e-12)  # Agreement of the two evaluations.


# The model comparison ranks a better fitting model first.
# Arguments:
#   none.
# Returns:
#   None: the test asserts the expected ranking.
def test_model_comparison_ranking():
    rng = np.random.default_rng(31)  # Deterministic generator of the compared contributions.
    better = -0.5 * rng.random((80, 40))  # Contributions of the better fitting model.
    worse = better - 0.4  # Contributions of the model that fits the points less well.
    records = compare_models({"better": better, "worse": worse})  # Leave one out comparison.
    assert ranking(records) == ["better", "worse"]  # The better fitting model ranks first.
    blocks = block_indices(np.linspace(0.0, 120.0, 40), 30.0)  # Contiguous blocks of the fixes.
    assert int(np.unique(blocks).size) == 5  # Four full blocks and the final boundary block.
