# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Construction of the study area of one tracked individual. The area
# is the region enclosing all retained fixes plus a buffer whose width scales
# with the square root of the estimated home range area, discretised on a
# Cartesian grid with cells outside the area masked.
# Manuscript: Section 4.5, inference for the wolf data.
# Inputs: projected fixes in metres and a buffer width. Outputs: the mask, the
# grid coordinates and the area of the study region.

import numpy as np  # Numerical arrays and elementary functions.
from shapely import contains_xy  # Vectorised point in polygon test of Shapely.
from shapely.geometry import MultiPoint  # Convex hull of the projected fixes.


# Buffer width implied by the estimated home range area.
# Arguments:
#   home_range_area (float): the ninety five per cent home range area in m^2.
#   factor (float): the dimensionless factor c of Section 4.5.
# Returns:
#   float: the buffer width b = c sqrt(A95) in metres.
def buffer_width(home_range_area, factor):
    return float(factor) * float(np.sqrt(float(home_range_area)))  # Buffer of Section 4.5.


# Build the masked Cartesian grid of the study area of one individual.
# Arguments:
#   easting (numpy.ndarray): projected first coordinate of the fixes, in metres.
#   northing (numpy.ndarray): projected second coordinate of the fixes, in metres.
#   buffer_metres (float): width of the buffer around the enclosing region.
#   cell (float): uniform cell width of the Cartesian grid, in metres.
# Returns:
#   dict: the boolean mask, the coordinates of the cell centres along each axis,
#   the polygon of the study area and its area in square metres.
def build_study_area(easting, northing, buffer_metres, cell):
    x_values = np.asarray(easting, dtype=float)  # Projected first coordinate of the fixes.
    y_values = np.asarray(northing, dtype=float)  # Projected second coordinate of the fixes.
    if x_values.size < 3:  # A convex hull requires at least three distinct points.
        raise ValueError("At least three fixes are required to build a study area")  # Reject.
    hull = MultiPoint(np.column_stack([x_values, y_values])).convex_hull  # Enclosing region.
    polygon = hull.buffer(float(buffer_metres))  # Study area with the buffer of Section 4.5.
    bounds = polygon.bounds  # Bounding box of the buffered study area.
    n_cols = int(np.ceil((bounds[2] - bounds[0]) / float(cell))) + 1  # Columns of the grid.
    n_rows = int(np.ceil((bounds[3] - bounds[1]) / float(cell))) + 1  # Rows of the grid.
    x_centres = bounds[0] + (np.arange(n_cols) + 0.5) * float(cell)  # Centres along the first axis.
    y_centres = bounds[1] + (np.arange(n_rows) + 0.5) * float(cell)  # Centres along the second axis.
    grid_x, grid_y = np.meshgrid(x_centres, y_centres, indexing="xy")  # Coordinates of the cells.
    mask = contains_xy(polygon, grid_x, grid_y)  # Cells whose centre lies inside the study area.
    # Assemble the mask together with the geometry that produced it.
    return {
        "mask": np.asarray(mask, dtype=bool),  # Boolean mask of the study area.
        "x_centres": x_centres,  # Coordinates of the cell centres along the first axis.
        "y_centres": y_centres,  # Coordinates of the cell centres along the second axis.
        "cell": float(cell),  # Uniform cell width of the Cartesian grid.
        "polygon": polygon,  # Buffered study area as a Shapely polygon.
        "area": float(np.sum(mask)) * float(cell) ** 2,  # Discretised area in square metres.
        "bounds": tuple(float(value) for value in bounds),  # Bounding box of the study area.
    }


# Indices of the grid cells that contain a set of points.
# Arguments:
#   x_values (numpy.ndarray): first coordinate of the points, in metres.
#   y_values (numpy.ndarray): second coordinate of the points, in metres.
#   area (dict): the mapping produced by build_study_area.
# Returns:
#   tuple: the row indices and the column indices of the containing cells.
def cell_indices(x_values, y_values, area):
    x_centres = area["x_centres"]  # Coordinates of the cell centres along the first axis.
    y_centres = area["y_centres"]  # Coordinates of the cell centres along the second axis.
    cell = area["cell"]  # Uniform cell width of the Cartesian grid.
    # Column index of the cell whose centre is nearest to each point.
    columns = np.floor((np.asarray(x_values, dtype=float) - x_centres[0]) / cell + 0.5).astype(int)
    # Row index of the cell whose centre is nearest to each point.
    rows = np.floor((np.asarray(y_values, dtype=float) - y_centres[0]) / cell + 0.5).astype(int)
    columns = np.clip(columns, 0, x_centres.size - 1)  # Keep the indices inside the grid.
    rows = np.clip(rows, 0, y_centres.size - 1)  # Keep the indices inside the grid.
    return rows, columns  # Row and column indices of the cells that contain the points.
