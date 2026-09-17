# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Autocorrelated kernel density estimation of the utilisation
# distribution of each tracked individual, following Fleming and co-authors
# (2015). The script reports the ninety five per cent home range area and the
# effective sample size for area used by the inclusion rule of Section 4.5.
# Manuscript: Section 4.5; Experiment 4.1 and Table 7.
# Inputs: a comma separated table with the columns individual_id,
# timestamp_utc, easting_m and northing_m, written by exp_4_1_wolf_preprocess.
# Outputs: a summary table with one row per individual and one raster of the
# utilisation distribution per individual.

# Command line arguments supplied by the calling Python script.
arguments <- commandArgs(trailingOnly = TRUE)  # Input file, output directory and grid size.
# Reject an invocation that does not supply the three required arguments.
if (length(arguments) < 3) {  # The script needs an input, an output and a grid size.
  # Abort with a usage message rather than failing later with an unclear error.
  stop("Usage: Rscript akde_ctmm.R <cleaned_fixes.csv> <output_directory> <grid_size>")
}  # End of the argument check.
# Path of the cleaned table of location fixes.
input_path <- arguments[1]  # Comma separated table written by the preprocessing script.
# Directory that receives the summary table and the utilisation distributions.
output_dir <- arguments[2]  # Output directory of Experiment 4.1.
# Number of grid cells along each axis of the utilisation distribution raster.
grid_size <- as.integer(arguments[3])  # Resolution of the exported raster.
# Load the continuous time movement modelling package.
suppressPackageStartupMessages(library(ctmm))  # Autocorrelated kernel density estimation.
# Create the output directory when it does not exist yet.
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)  # Recursive creation.
# Read the cleaned table of location fixes.
fixes <- read.csv(input_path, stringsAsFactors = FALSE)  # Cleaned and projected fixes.
# Identifiers of the individuals present in the cleaned table.
individuals <- unique(fixes$individual_id)  # One entry per tracked animal.
# Accumulator for the summary rows of all individuals.
summary_rows <- list()  # Filled with one record per individual.
# Fit a movement model and an estimator to each individual in turn.
for (identifier in individuals) {  # Loop over the tracked animals.
  # Rows of the cleaned table that belong to the current individual.
  subset_rows <- fixes[fixes$individual_id == identifier, ]  # Fixes of this animal.
  # Table in the column layout expected by the movement modelling package.
  telemetry_frame <- data.frame(
    timestamp = as.POSIXct(subset_rows$timestamp_utc, tz = "UTC"),  # Time of each fix.
    x = subset_rows$easting_m,  # Projected first coordinate, in metres.
    y = subset_rows$northing_m,  # Projected second coordinate, in metres.
    individual.local.identifier = as.character(identifier)  # Identifier of the animal.
  )  # End of the telemetry table.
  # Convert the table into the telemetry object of the package.
  telemetry_object <- as.telemetry(telemetry_frame, projection = NULL)  # Telemetry object.
  # Empirical variogram used to initialise the movement model.
  variogram_estimate <- variogram(telemetry_object)  # Empirical semivariance of the track.
  # Initial guess of the continuous time movement model.
  initial_guess <- ctmm.guess(telemetry_object, variogram = variogram_estimate, interactive = FALSE)
  # Model selection among the candidate continuous time movement models.
  fitted_model <- ctmm.select(telemetry_object, CTMM = initial_guess, verbose = FALSE)
  # Autocorrelated kernel density estimate of the utilisation distribution.
  density_estimate <- akde(telemetry_object, fitted_model, grid = list(dr.fn = min, res = grid_size))
  # Summary of the home range area, reported in square kilometres.
  area_summary <- summary(density_estimate, units = FALSE)  # Area in square metres.
  # Point estimate of the ninety five per cent home range area.
  area_estimate <- area_summary$CI[1, 2]  # Central column of the confidence interval.
  # Effective sample size for area reported by the fitted model.
  effective_size <- summary(fitted_model, units = FALSE)$DOF["area"]  # Degrees of freedom.
  # Export the utilisation distribution as a comma separated raster.
  raster_values <- raster(density_estimate, DF = "PDF")  # Probability density raster.
  # Path of the exported raster of the current individual.
  raster_path <- file.path(output_dir, paste0("ud_", identifier, ".csv"))  # Output path.
  # Write the raster as a plain table of coordinates and density values.
  write.csv(as.data.frame(raster_values, xy = TRUE), raster_path, row.names = FALSE)
  # Record the summary row of the current individual.
  summary_rows[[length(summary_rows) + 1]] <- data.frame(
    individual_id = as.character(identifier),  # Identifier of the animal.
    n_fixes = nrow(subset_rows),  # Number of retained fixes of the animal.
    area_m2 = as.numeric(area_estimate),  # Ninety five per cent home range area.
    n_eff = as.numeric(effective_size),  # Effective sample size for area.
    ud_path = raster_path  # Path of the exported utilisation distribution.
  )  # End of the summary row.
}  # End of the loop over the tracked animals.
# Combine the summary rows into a single table.
summary_table <- do.call(rbind, summary_rows)  # One row per tracked animal.
# Write the summary table into the output directory.
write.csv(summary_table, file.path(output_dir, "akde_summary.csv"), row.names = FALSE)
# Report completion so that the calling Python script can log it.
cat("akde_ctmm.R completed for", length(individuals), "individuals\n")  # Completion message.
