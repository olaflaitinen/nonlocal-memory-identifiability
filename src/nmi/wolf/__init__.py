# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Package marker for the wolf application of Section 6, which fits
# model (1) to the publicly archived Global Positioning System location data of
# Latham and Boutin (2019).
# Manuscript: Sections 4.5 and 6.
# Inputs: none. Outputs: the namespace of the wolf application.

# Modules of the wolf application, imported explicitly by the experiment scripts.
__all__ = ["schema", "study_area", "masked_solver", "point_likelihood", "model_comparison"]
