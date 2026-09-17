# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Install the R packages required by the autocorrelated kernel density
# estimation of Section 4.5. Only Experiment 4.1 depends on R.
# Manuscript: Section 4.5, inference for the wolf data.
# Inputs: none. Outputs: the packages ctmm and its dependencies, installed into
# the first writable library of the active R installation.

# Packages required by the autocorrelated kernel density estimation.
required <- c("ctmm", "sp")  # Continuous time movement models and spatial classes.
# Repository from which the packages are downloaded.
repository <- "https://cloud.r-project.org"  # Comprehensive R Archive Network mirror.
# Packages that are not yet available in the active R library.
missing <- required[!(required %in% rownames(installed.packages()))]  # Absent packages only.
# Install the absent packages, if any remain.
if (length(missing) > 0) {  # At least one required package is absent.
  install.packages(missing, repos = repository)  # Download and install the absent packages.
}  # End of the conditional installation.
# Report the installed versions so that the environment can be recorded.
for (name in required) {  # Print one line per required package.
  cat(name, as.character(packageVersion(name)), "\n")  # Package name and installed version.
}  # End of the version report.
