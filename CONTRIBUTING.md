# Contributing

This repository accompanies a manuscript, so its contents are expected to match
the methods described there. Contributions that change the mathematical model,
the parameter values, the experimental design or the statistical thresholds are
not accepted unless the manuscript changes with them.

## Coding standards

The full standards are recorded in `docs/coding_standards.md`. In summary:

- Every source and configuration file that supports comments opens with the
  three line notice of the Mozilla Public License 2.0, followed by a comment
  block that states the purpose of the file, the manuscript sections it
  supports and its inputs and outputs.
- All comments use the character `#`. Python files contain no docstrings.
- Every function and class is preceded by a comment block that states its
  purpose, its arguments with their types and meanings, its return value and
  the manuscript result it implements.
- Every non-blank line of executable code, configuration or shell script
  carries an explanatory comment, inline or on the line immediately above.
- Comments must be accurate. When the code changes, the comments change too.

## Character policy

- The em dash (U+2014) and the en dash (U+2013) must not appear anywhere.
- Emoji and other pictographic symbols must not appear anywhere.
- Source and configuration files contain ASCII characters only. Markdown
  documentation may use mathematical Unicode symbols such as Greek letters.
- Prose uses British academic English and a professional tone.

## Checks before a commit

```bash
make policy     # character, comment and licence header policies
make lint       # ruff with a line length of 120 characters
make test       # pytest without the slow tests
make test-all   # pytest including the slow tests
```

A change is ready only when all four commands pass. The same commands run in
the continuous integration workflow on every push to `main`.

## Adding an experiment

1. Add a configuration file under `configs/` that names its base configuration
   and overrides only the entries that the experiment changes.
2. Add a script under `experiments/` that accepts `--config` and, where
   relevant, `--condition-index`, `--chain-index`, `--output-dir` and
   `--resume`.
3. Write raw output under `results/raw/<experiment>/`, which is untracked, and
   a compact summary under `results/summary/`, which is tracked.
4. Add a row to the mapping table in `docs/experiments.md`.
5. Add a Slurm job file under `slurm/` when the experiment needs a cluster.

## Commit messages

Write concise imperative sentences in plain English that describe one logical
unit of work, for example `Add pseudo-spectral solver for model (1)` or
`Fix dealiasing mask for odd grid sizes`. Commit messages carry no trailers, no
emoji and none of the dash characters forbidden above. Keep commits atomic: one
file, or a small group of files that only make sense together.
