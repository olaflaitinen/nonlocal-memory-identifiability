# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Enforce the comment policy of the repository. Every non-blank line of
# executable code, configuration or shell script must carry an explanatory
# comment introduced by the character "#", either inline on the same line or on
# the line immediately above. Python files must not contain docstrings, and
# every Python function or class must be preceded by a comment block.
# Manuscript: supports the coding standards recorded in docs/coding_standards.md.
# Inputs: the tracked files of the git working tree, or explicit paths given on
# the command line.
# Outputs: a report on standard output and a non-zero exit status on violation.

import io  # Provides an in-memory byte stream for the tokenizer.
import subprocess  # Used to obtain the list of files tracked by git.
import sys  # Used for command line arguments and the process exit status.
import tokenize  # Provides a faithful tokenizer for Python source files.
from pathlib import Path  # Portable filesystem paths.

# Suffixes of files whose comments are checked with the line based parser.
LINE_COMMENT_SUFFIXES = {
    ".sh",  # Shell helper scripts.
    ".sbatch",  # Slurm batch scripts, which are shell scripts with directives.
    ".R",  # R scripts of the wolf application.
    ".yaml",  # Experiment configurations.
    ".yml",  # Continuous integration workflow.
    ".toml",  # Packaging metadata.
    ".cff",  # Citation metadata, which is a YAML document.
    ".gitignore",  # Ignore rules, treated as a configuration file.
}  # End of the line based suffix set.

# File names without a suffix whose comments are checked with the line parser.
LINE_COMMENT_NAMES = {
    "Makefile",  # The build driver, in which recipes are shell commands.
    ".gitignore",  # Ignore rules when matched by name rather than suffix.
}  # End of the line based name set.

# Files that cannot carry comments and are therefore exempt from the policy.
EXEMPT_PATHS = {
    ".zenodo.json",  # JSON has no comment syntax.
    "LICENSE",  # Supplied by the licence template and never modified.
    "data/checksums.sha256",  # A checksum manifest consumed by sha256sum.
}  # End of the exempt path set.

# Tokens that may terminate a bracketed continuation without their own comment.
CONTINUATION_CHARACTERS = set(")]}:,\\")  # Closing brackets and separators only.


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


# Decide whether a stripped line consists only of continuation punctuation.
# Arguments:
#   stripped (str): the line with leading and trailing whitespace removed.
# Returns:
#   bool: True when the line only closes a multi-line expression.
def is_pure_continuation(stripped):
    if not stripped:  # An empty line carries no code at all.
        return True  # Blank lines never require a comment.
    return all(char in CONTINUATION_CHARACTERS for char in stripped)  # Punctuation only.


# Check the comment policy of a single Python source file.
# Arguments:
#   path (pathlib.Path): path of the Python file to inspect.
#   text (str): full contents of the file.
# Returns:
#   list: human readable descriptions of the violations found.
def check_python(path, text):
    problems = []  # Accumulator for the violations of this file.
    lines = text.splitlines()  # Physical lines used for the per-line report.
    commented = set()  # Line numbers that carry an inline or standalone comment.
    comment_only = set()  # Line numbers that consist of a comment alone.
    code_lines = set()  # Line numbers on which at least one code token begins.
    depth_at_line = {}  # Bracket nesting depth observed at the start of each line.
    stream = io.BytesIO(text.encode("utf-8")).readline  # Byte reader for the tokenizer.
    depth = 0  # Running bracket nesting depth across the token stream.
    previous_row = 0  # Row of the previously seen significant token.
    try:  # Guard against files that are not syntactically valid Python.
        tokens = list(tokenize.tokenize(stream))  # Materialise the whole token stream.
    except (tokenize.TokenError, SyntaxError, IndentationError) as error:  # Malformed source.
        problems.append(f"{path}: could not tokenise the file ({error})")  # Report the failure.
        return problems  # Abandon the inspection of this file.
    for token in tokens:  # Walk every token of the file in source order.
        row = token.start[0]  # Line number on which the token begins.
        if row not in depth_at_line:  # First token seen on this line.
            depth_at_line[row] = depth  # Record the nesting depth at the line start.
        if token.type == tokenize.COMMENT:  # A comment token, inline or standalone.
            commented.add(row)  # The line carries an explanatory comment.
            if lines[row - 1].strip().startswith("#"):  # Nothing precedes the comment.
                comment_only.add(row)  # The line is a standalone comment line.
            continue  # Comments themselves never require a further comment.
        # Layout tokens describe the shape of the source rather than its content.
        if token.type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            continue  # Layout tokens carry no code and are ignored here.
        if token.type in (tokenize.ENCODING, tokenize.ENDMARKER):  # Stream markers.
            continue  # Stream markers carry no code and are ignored here.
        code_lines.add(row)  # The line contains at least one genuine code token.
        if token.type == tokenize.OP:  # Track the bracket nesting depth.
            if token.string in "([{":  # An opening bracket increases the depth.
                depth += 1  # Record the deeper nesting level.
            elif token.string in ")]}":  # A closing bracket decreases the depth.
                depth = max(0, depth - 1)  # Never allow a negative nesting depth.
        if token.type == tokenize.STRING and previous_row == 0:  # A leading string literal.
            problems.append(f"{path}:{row}: module docstring found, use comment blocks instead")  # Report.
        previous_row = row  # Remember the row of the last significant token.
    for row in sorted(code_lines):  # Inspect every line that contains code.
        if row in commented:  # The line carries its own inline comment.
            continue  # The policy is satisfied for this line.
        if (row - 1) in comment_only:  # The preceding line is a standalone comment.
            continue  # The policy is satisfied for this line.
        stripped = lines[row - 1].strip()  # Textual content of the offending line.
        if depth_at_line.get(row, 0) > 0 and is_pure_continuation(stripped):  # Closing bracket.
            continue  # A bracket that only terminates an expression is exempt.
        problems.append(f"{path}:{row}: code line without an explanatory comment: {stripped}")  # Report.
    problems.extend(check_python_structure(path, tokens, lines, comment_only))  # Structural checks.
    return problems  # Return every violation found in this file.


# Check that Python definitions are documented and that no docstring is present.
# Arguments:
#   path (pathlib.Path): path of the Python file to inspect.
#   tokens (list): the token stream produced by the tokenize module.
#   lines (list): the physical lines of the file.
#   comment_only (set): line numbers that consist of a comment alone.
# Returns:
#   list: human readable descriptions of the violations found.
def check_python_structure(path, tokens, lines, comment_only):
    problems = []  # Accumulator for the structural violations of this file.
    expect_docstring = False  # True while the next statement could be a docstring.
    for index, token in enumerate(tokens):  # Walk the token stream with its position.
        if token.type == tokenize.NAME and token.string in ("def", "class"):  # A definition.
            row = token.start[0]  # Line number of the definition keyword.
            if row > 1 and lines[row - 2].strip().startswith("@"):  # A decorator precedes it.
                continue  # The decorator line carries the comment requirement instead.
            if (row - 1) not in comment_only:  # No comment block precedes the definition.
                name = lines[row - 1].strip()  # Textual content of the definition line.
                problems.append(f"{path}:{row}: definition without a preceding comment block: {name}")  # Report.
            expect_docstring = True  # The body of the definition may open with a docstring.
            continue  # Continue with the next token of the stream.
        if expect_docstring and token.type == tokenize.STRING:  # A string opens the body.
            previous = tokens[index - 1]  # Token immediately preceding the string literal.
            if previous.type in (tokenize.INDENT, tokenize.NEWLINE, tokenize.NL):  # Statement start.
                problems.append(f"{path}:{token.start[0]}: docstring found, use comment blocks instead")  # Report.
            expect_docstring = False  # Only the first statement of a body can be a docstring.
            continue  # Continue with the next token of the stream.
        # Any token other than those listed below marks the start of the body.
        if expect_docstring and token.type not in (
            tokenize.NEWLINE,  # End of the definition header.
            tokenize.NL,  # Blank line inside the header or body.
            tokenize.INDENT,  # Indentation that opens the body.
            tokenize.COMMENT,  # Comment lines inside the body.
            tokenize.OP,  # Punctuation of the definition header.
            tokenize.NAME,  # Argument names of the definition header.
            tokenize.NUMBER,  # Numeric defaults of the definition header.
        ):  # Any other token means that the body has begun with real code.
            expect_docstring = False  # Stop looking for a docstring in this definition.
    return problems  # Return every structural violation found in this file.


# Remove quoted material from a line so that comment markers can be located.
# Arguments:
#   line (str): a single physical line of a shell, YAML, TOML or R file.
# Returns:
#   str: the line with the contents of quoted strings replaced by spaces.
def blank_quoted_regions(line):
    output = []  # Characters of the sanitised line.
    quote = None  # The quote character that currently delimits a string, if any.
    for char in line:  # Walk the characters of the line in order.
        if quote is None and char in ("'", '"'):  # A string literal opens here.
            quote = char  # Remember which quote character must close it.
            output.append(" ")  # Replace the quote itself by a space.
            continue  # Continue with the next character of the line.
        if quote is not None:  # The scanner is inside a string literal.
            if char == quote:  # The matching quote closes the literal.
                quote = None  # Leave the string literal.
            output.append(" ")  # Replace the quoted character by a space.
            continue  # Continue with the next character of the line.
        output.append(char)  # Characters outside string literals are preserved.
    return "".join(output)  # Reassemble the sanitised line.


# Check the comment policy of a file that uses line comments introduced by "#".
# Arguments:
#   path (pathlib.Path): path of the file to inspect.
#   text (str): full contents of the file.
# Returns:
#   list: human readable descriptions of the violations found.
def check_line_comments(path, text):
    problems = []  # Accumulator for the violations of this file.
    lines = text.splitlines()  # Physical lines of the file.
    previous_is_comment = False  # True when the previous line was a comment line.
    for number, line in enumerate(lines, start=1):  # Walk the lines of the file.
        stripped = line.strip()  # Textual content without surrounding whitespace.
        if not stripped:  # A blank line separates logical units.
            previous_is_comment = False  # A blank line cancels a preceding comment.
            continue  # Blank lines never require a comment.
        if stripped.startswith("#"):  # The whole line is a comment.
            previous_is_comment = True  # The next code line is covered by this comment.
            continue  # Comment lines satisfy the policy by construction.
        if stripped.startswith("---") or stripped == "...":  # YAML document markers.
            previous_is_comment = False  # Document markers carry no comment forward.
            continue  # Structural markers of a YAML stream need no comment.
        if "#" in blank_quoted_regions(line):  # An inline comment follows the code.
            previous_is_comment = False  # The comment belongs to this line only.
            continue  # The policy is satisfied for this line.
        if previous_is_comment:  # The preceding line was a standalone comment.
            previous_is_comment = False  # The comment covers exactly one code line.
            continue  # The policy is satisfied for this line.
        if is_pure_continuation(stripped):  # Only closing brackets or separators.
            continue  # The comment of the opening line covers this terminator.
        problems.append(f"{path}:{number}: line without an explanatory comment: {stripped}")  # Report.
    return problems  # Return every violation found in this file.


# Inspect one file according to its type.
# Arguments:
#   path (pathlib.Path): path of the file to inspect.
# Returns:
#   list: human readable descriptions of the violations found.
def check_file(path):
    if str(path) in EXEMPT_PATHS:  # Files that cannot carry comments at all.
        return []  # Skip the file without inspecting it.
    if not path.is_file():  # Deleted or unavailable files are silently ignored.
        return []  # Nothing to inspect.
    is_python = path.suffix == ".py"  # Python sources use the tokenizer based check.
    is_line = path.suffix in LINE_COMMENT_SUFFIXES or path.name in LINE_COMMENT_NAMES  # Others.
    if not (is_python or is_line):  # The file type carries no comment requirement.
        return []  # Markdown, data and binary files are outside the policy.
    text = path.read_text(encoding="utf-8")  # Read the whole file as text.
    if is_python:  # Dispatch to the Python specific check.
        return check_python(path, text)  # Tokenizer based inspection.
    return check_line_comments(path, text)  # Line based inspection for all other types.


# Entry point of the comment policy check.
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
        print(f"comment policy: {len(problems)} violation(s)")  # Summarise the outcome.
        return 1  # Signal failure to the caller.
    print("comment policy: no violations")  # Summarise a successful run.
    return 0  # Signal success to the caller.


if __name__ == "__main__":  # Allow the module to be used as a command line tool.
    sys.exit(main(sys.argv[1:]))  # Run the check and propagate the exit status.
