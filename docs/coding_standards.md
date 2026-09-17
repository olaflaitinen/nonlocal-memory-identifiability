# Coding standards

These standards are enforced by the three scripts under `tools/` and by the
continuous integration workflow. A change that violates them fails the build.

## Character policy

1. The em dash (U+2014) and the en dash (U+2013) must not appear in any file of
   the repository. Use a comma, a colon, parentheses, a full stop, or the ASCII
   hyphen-minus instead.
2. Emoji and other pictographic symbols must not appear anywhere, including
   file contents, commit messages, workflow names and log messages.
3. Source and configuration files (`.py`, `.R`, `.sh`, `.sbatch`, `.yaml`,
   `.yml`, `.toml`, `.cff`, `.gitignore`, `Makefile`) contain ASCII characters
   only. Markdown documentation may contain mathematical Unicode symbols such
   as Greek letters, but it is still subject to rules 1 and 2.
4. The documentation and the code comments use British academic English, for
   example "nondimensionalise", "organised" and "behaviour".
5. The tone is professional and academic. There is no informal language, no
   marketing language and no exclamation mark.

Enforced by `tools/check_text_policy.py`.

## Comment policy

1. All comments are written with the character `#`. Python files contain no
   docstrings; modules, functions and classes are documented with `#` comment
   blocks instead.
2. Every file opens with the licence notice of Section "Licence" below,
   followed by a `#` header block that states the purpose of the file, the
   manuscript sections, equations, tables or figures that it supports, and its
   inputs and outputs.
3. Every function and class is preceded by a `#` block that states its purpose,
   each argument with its type and meaning, the return value, and the
   manuscript equation or result that it implements.
4. Every non-blank line of executable code, configuration or shell script
   carries an explanatory `#` comment, either inline on the same line or on the
   line immediately above. The comment states the purpose of that line in
   academic terms, not a paraphrase of the syntax. A line that only closes a
   multi-line expression is covered by the comment of the line that opens it.
5. Comments must be accurate. When the code changes, the comments change with
   it.
6. The only files exempt from this policy are those whose syntax has no comment
   form: `.zenodo.json`, `LICENSE` and `data/checksums.sha256`. They are listed
   explicitly in `tools/check_comment_policy.py`.

Enforced by `tools/check_comment_policy.py`.

## Licence

The project licence is the Mozilla Public License 2.0, with the SPDX identifier
`MPL-2.0`. The `LICENSE` file comes from the GitHub licence template and is
never edited. Every source and configuration file that supports comments opens
with exactly these three lines:

```
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
```

Enforced by `tools/check_license_headers.py`.

## Linting and tests

The linter is ruff with a line length of 120 characters, configured in
`pyproject.toml`. The test runner is pytest, and tests that run a full forward
solve or a sampler carry the `slow` marker.

```bash
make policy     # the three checks above
make lint       # ruff
make test       # pytest without the slow tests
make test-all   # pytest including the slow tests
```

## Commit messages

Commit messages are concise imperative sentences in plain English that describe
one logical unit of work, for example `Add pseudo-spectral solver for model (1)`
or `Fix dealiasing mask for odd grid sizes`. They carry no trailers, no emoji
and none of the dash characters forbidden above.
