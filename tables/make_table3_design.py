# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Table 3 of the manuscript, listing the critical aggregation ratio,
# the aggregation ratio, the combined advection strength and the index m_star
# for each detection kernel and perceptual range of the synthetic design.
# Manuscript: Section 4.2, Table 3; Proposition 3 and Corollary 1.
# Inputs: the design configuration. Outputs: results/tables/Table3.csv and
# results/tables/Table3.tex.

import argparse  # Command line interface of the table script.
import sys  # Process exit status of the table script.

from nmi.config import load_experiment_config  # Configuration loader with base inheritance.
from nmi.design import table3_values  # Parameter values of Table 3.
from nmi.io import ensure_dir, write_csv, write_latex  # Output writers of the table script.

# Caption of the table, reproduced in the LaTeX output.
CAPTION = "Parameter values and prior bounds used in the synthetic experiments"  # Table 3 caption.


# Parse the command line arguments of the script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   argparse.Namespace: the parsed arguments.
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Table 3")  # Argument parser of the script.
    parser.add_argument("--config", default="configs/design_1d.yaml", help="configuration")  # Input.
    parser.add_argument("--output-dir", default="results/tables", help="output directory")  # Output.
    return parser.parse_args(argv)  # Parsed command line arguments of the script.


# Entry point of the table script.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero on success.
def main(argv):
    arguments = parse_arguments(argv)  # Parsed command line arguments of the script.
    config = load_experiment_config(arguments.config)  # Merged and validated configuration.
    design = config["design"]  # Block that describes the synthetic design.
    records = []  # Accumulator for the rows of Table 3.
    for kernel in ("tophat", "gaussian"):  # Both detection kernel families of the manuscript.
        for radius in sorted(float(value) for value in design["perceptual_ranges"]):  # Ranges.
            values = table3_values(kernel, radius, design)  # Parameter values of this row.
            records.append(  # One row of Table 3.
                {  # Values of this combination of kernel family and perceptual range.
                    "kernel": "Top-hat" if kernel == "tophat" else "Gaussian",  # Kernel family.
                    "radius": f"{radius:.3f}",  # Perceptual range R of the row.
                    "critical_kappa": f"{values['critical_kappa']:.3f}",  # Critical ratio.
                    "aggregation_ratio": f"{values['aggregation_ratio']:.3f}",  # Ratio kappa.
                    "advection": f"{values['advection']:.5f}",  # Combined advection strength.
                    # The index of Corollary 1 exists only for the top-hat kernel.
                    "m_star": str(values["m_star"]) if values["m_star"] else "Not applicable",
                }
            )  # Row appended to Table 3.
    # Column names of the output, in the order used by the manuscript.
    columns = ["kernel", "radius", "critical_kappa", "aggregation_ratio", "advection", "m_star"]
    # Column headers of the typeset table, written in mathematical notation.
    headers = ["Kernel", "$R$", "$\\kappa_c(R)$", "$\\kappa$", "$\\gamma$", "$m^*$"]
    directory = ensure_dir(arguments.output_dir)  # Directory that receives the table files.
    csv_path = write_csv(directory / "Table3.csv", records, columns)  # Comma separated output.
    # LaTeX output in the booktabs style expected by the journal template.
    tex_path = write_latex(directory / "Table3.tex", records, columns, headers, CAPTION, "tab:design")
    print(f"wrote {csv_path} and {tex_path} with {len(records)} row(s)")  # Report the outputs.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line script.
    sys.exit(main(sys.argv[1:]))  # Build the table and propagate the exit status.
