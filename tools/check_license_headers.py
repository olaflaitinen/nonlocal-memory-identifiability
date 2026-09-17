# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Enforce the licence header policy of the repository. Every tracked
# file whose syntax supports comments must begin with the three line notice of
# the Mozilla Public License, version 2.0.
# Manuscript: supports the coding standards recorded in docs/coding_standards.md.
# Inputs: the tracked files of the git working tree, or explicit paths given on
# the command line.
# Outputs: a report on standard output and a non-zero exit status on violation.

import subprocess  # Used to obtain the list of files tracked by git.
import sys  # Used for command line arguments and the process exit status.
from pathlib import Path  # Portable filesystem paths.

# The three lines that must open every file that supports comments.
HEADER_LINES = (
    "# This Source Code Form is subject to the terms of the Mozilla Public",  # First line.
    "# License, v. 2.0. If a copy of the MPL was not distributed with this",  # Second line.
    "# file, You can obtain one at https://mozilla.org/MPL/2.0/.",  # Third line.
)  # End of the required header.

# Suffixes of files that must carry the licence header.
HEADER_SUFFIXES = {
    ".py",  # Python modules and scripts.
    ".R",  # R scripts of the wolf application.
    ".sh",  # Shell helper scripts.
    ".sbatch",  # Slurm batch scripts.
    ".yaml",  # Experiment configurations.
    ".yml",  # Continuous integration workflow.
    ".toml",  # Packaging metadata.
    ".cff",  # Citation metadata.
    ".gitignore",  # Ignore rules, treated as a configuration file.
}  # End of the suffix set that requires a header.

# File names without a suffix that must carry the licence header.
HEADER_NAMES = {
    "Makefile",  # The build driver.
    ".gitignore",  # Ignore rules when matched by name rather than suffix.
}  # End of the name set that requires a header.

# Files that cannot carry comments and are therefore exempt from the policy.
EXEMPT_PATHS = {
    ".zenodo.json",  # JSON has no comment syntax.
    "LICENSE",  # Supplied by the licence template and never modified.
    "data/checksums.sha256",  # A checksum manifest consumed by sha256sum.
}  # End of the exempt path set.


# Collect the files that the policy applies to.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   list: paths of the files to inspect, relative to the repository root.
def collect_files(argv):
    if argv:  # Explicit paths were supplied by the caller.
        return [Path(item) for item in argv]  # Inspect exactly those paths.
    command = ["git", "ls-files"]  # List every file tracked by the repository.
    output = subprocess.run(command, capture_output=True, text=True, check=True)  # Run git.
    names = [line for line in output.stdout.splitlines() if line]  # Drop empty lines.
    return [Path(name) for name in names]  # Convert the names to path objects.


# Test whether a given path must carry the licence header.
# Arguments:
#   path (pathlib.Path): path of the file under test.
# Returns:
#   bool: True when the file must open with the three line notice.
def requires_header(path):
    if str(path) in EXEMPT_PATHS:  # Files that cannot carry comments at all.
        return False  # No header is required or possible.
    if path.name in HEADER_NAMES:  # Files identified by their exact name.
        return True  # The Makefile and the ignore file belong to this class.
    return path.suffix in HEADER_SUFFIXES  # Otherwise decide by the suffix.


# Inspect one file and return the violations that it contains.
# Arguments:
#   path (pathlib.Path): path of the file to inspect.
# Returns:
#   list: human readable descriptions of the violations found.
def check_file(path):
    if not requires_header(path):  # The file type carries no header requirement.
        return []  # Nothing to inspect.
    if not path.is_file():  # Deleted or unavailable files are silently ignored.
        return []  # Nothing to inspect.
    text = path.read_text(encoding="utf-8")  # Read the whole file as text.
    lines = text.splitlines()  # Physical lines of the file.
    if len(lines) < len(HEADER_LINES):  # The file is shorter than the notice.
        return [f"{path}:1: missing the Mozilla Public License 2.0 notice"]  # Report the omission.
    for index, expected in enumerate(HEADER_LINES):  # Compare the opening lines.
        if lines[index] != expected:  # The line differs from the required text.
            return [f"{path}:{index + 1}: licence notice line differs from the required text"]  # Report.
    return []  # The header is present and exact.


# Entry point of the licence header check.
# Arguments:
#   argv (list): command line arguments after the program name.
# Returns:
#   int: zero when the policy holds and one otherwise.
def main(argv):
    problems = []  # Accumulator for the violations of the whole repository.
    for path in collect_files(argv):  # Inspect every file in scope.
        problems.extend(check_file(path))  # Append the violations of that file.
    for problem in problems:  # Report the violations in the order found.
        print(problem)  # Write one violation per line of standard output.
    if problems:  # At least one violation was detected.
        print(f"licence headers: {len(problems)} violation(s)")  # Summarise the outcome.
        return 1  # Signal failure to the caller.
    print("licence headers: no violations")  # Summarise a successful run.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line tool.
    sys.exit(main(sys.argv[1:]))  # Run the check and propagate the exit status.
