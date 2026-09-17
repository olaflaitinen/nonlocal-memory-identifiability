# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Enforce the character policy of the repository on all tracked text
# files. The policy forbids the em dash (U+2014), the en dash (U+2013) and
# pictographic symbols anywhere, and forbids any non-ASCII character in source
# and configuration files. Markdown documentation may use mathematical Unicode.
# Manuscript: supports the coding standards recorded in docs/coding_standards.md.
# Inputs: the tracked files of the git working tree, or explicit paths given on
# the command line.
# Outputs: a report on standard output and a non-zero exit status on violation.

import subprocess  # Used to obtain the list of files tracked by git.
import sys  # Used for command line arguments and the process exit status.
import unicodedata  # Used to classify characters by Unicode category.
from pathlib import Path  # Portable filesystem paths.

# Code points that are forbidden in every tracked text file.
FORBIDDEN_CODEPOINTS = {
    0x2014: "em dash",  # U+2014, forbidden by the character policy.
    0x2013: "en dash",  # U+2013, forbidden by the character policy.
    0x2012: "figure dash",  # U+2012, a further dash form that is also excluded.
    0x2015: "horizontal bar",  # U+2015, a further dash form that is also excluded.
}  # End of the forbidden code point table.

# File suffixes that must contain ASCII characters only.
ASCII_ONLY_SUFFIXES = {
    ".py",  # Python modules and scripts.
    ".R",  # R scripts of the wolf application.
    ".sh",  # Shell helper scripts.
    ".sbatch",  # Slurm batch scripts.
    ".yaml",  # Experiment configurations.
    ".yml",  # Continuous integration workflow.
    ".toml",  # Packaging metadata.
    ".cff",  # Citation metadata.
    ".json",  # Archival metadata.
    ".sha256",  # Data checksum manifest.
    ".gitignore",  # Ignore rules, treated as a configuration file.
}  # End of the ASCII only suffix set.

# File names without a suffix that must contain ASCII characters only.
ASCII_ONLY_NAMES = {
    "Makefile",  # The build driver.
    ".gitignore",  # Ignore rules when matched by name rather than suffix.
}  # End of the ASCII only name set.

# File suffixes that are treated as binary and are therefore not inspected.
BINARY_SUFFIXES = {
    ".png",  # Raster images.
    ".eps",  # Encapsulated PostScript figures.
    ".pdf",  # Portable document format previews.
    ".h5",  # Hierarchical data format archives.
    ".npz",  # Compressed NumPy archives.
}  # End of the binary suffix set.

# Files that carry an upstream licence text and are excluded from the scan.
EXEMPT_PATHS = {
    "LICENSE",  # Supplied by the licence template and never modified.
}  # End of the exempt path set.


# Determine whether a character is a pictographic symbol such as an emoji.
# Arguments:
#   char (str): a single character.
# Returns:
#   bool: True when the character is pictographic and therefore forbidden.
def is_pictographic(char):
    code = ord(char)  # Numeric code point of the character under test.
    if code < 0x2100:  # Characters below this point are never emoji in practice.
        return False  # Latin, Greek and common punctuation are acceptable here.
    if unicodedata.category(char) == "So":  # Category "Symbol, other" covers emoji.
        return True  # Reject every pictographic symbol.
    if 0x1F000 <= code <= 0x1FAFF:  # Supplementary pictographic planes.
        return True  # Reject the supplementary emoji blocks explicitly.
    return code in (0xFE0F, 0x200D)  # Variation selector and zero width joiner are rejected.


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


# Test whether a given path must contain ASCII characters only.
# Arguments:
#   path (pathlib.Path): path of the file under test.
# Returns:
#   bool: True when the file is a source or configuration file.
def is_ascii_only(path):
    if path.name in ASCII_ONLY_NAMES:  # Files identified by their exact name.
        return True  # The Makefile and the ignore file belong to this class.
    return path.suffix in ASCII_ONLY_SUFFIXES  # Otherwise decide by the suffix.


# Inspect one file and return the violations that it contains.
# Arguments:
#   path (pathlib.Path): path of the file to inspect.
# Returns:
#   list: human readable descriptions of the violations found.
def check_file(path):
    problems = []  # Accumulator for the violations of this file.
    if str(path) in EXEMPT_PATHS:  # The licence text is supplied upstream.
        return problems  # Skip the file without inspecting it.
    if path.suffix in BINARY_SUFFIXES:  # Binary artefacts carry no policy text.
        return problems  # Skip binary files.
    if not path.is_file():  # Deleted or unavailable files are silently ignored.
        return problems  # Nothing to inspect.
    try:  # Guard against files that are not valid UTF-8 text.
        text = path.read_text(encoding="utf-8")  # Read the whole file as text.
    except UnicodeDecodeError:  # The file is binary in practice.
        return problems  # Binary content is outside the scope of the policy.
    ascii_only = is_ascii_only(path)  # Decide which rule set applies to this file.
    for number, line in enumerate(text.splitlines(), start=1):  # Walk the lines.
        for column, char in enumerate(line, start=1):  # Walk the characters.
            code = ord(char)  # Numeric code point of the current character.
            if code in FORBIDDEN_CODEPOINTS:  # Dash forms forbidden everywhere.
                name = FORBIDDEN_CODEPOINTS[code]  # Descriptive name of the character.
                problems.append(f"{path}:{number}:{column}: forbidden {name} (U+{code:04X})")  # Record it.
                continue  # Continue with the next character of the line.
            if is_pictographic(char):  # Emoji and other pictographic symbols.
                problems.append(f"{path}:{number}:{column}: forbidden pictographic symbol (U+{code:04X})")  # Record.
                continue  # Continue with the next character of the line.
            if ascii_only and code > 0x7F:  # Non-ASCII in a source or configuration file.
                where = f"{path}:{number}:{column}"  # Location of the offending character.
                problems.append(f"{where}: non-ASCII character (U+{code:04X}) in a source file")  # Record.
    return problems  # Return every violation found in this file.


# Entry point of the character policy check.
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
        print(f"character policy: {len(problems)} violation(s)")  # Summarise the outcome.
        return 1  # Signal failure to the caller.
    print("character policy: no violations")  # Summarise a successful run.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line tool.
    sys.exit(main(sys.argv[1:]))  # Run the check and propagate the exit status.
