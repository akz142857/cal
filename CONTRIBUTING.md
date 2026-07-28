# Contributing to Cal

Cal welcomes replication attempts, critical reviews, new controls, simulator
extensions, documentation improvements, and code contributions.

## Before opening a pull request

1. Read `RESEARCH_STATUS.md` and the relevant preregistration or stage report.
2. Open a Discussion for a new research direction or an Issue for a bounded,
   testable change.
3. Do not modify a frozen protocol, its checksum, a published result, or a
   source-locked implementation in place. Propose a new protocol version and
   explain the scientific reason.
4. Keep holdout observations out of design discussions. Publicly documented
   historical holdouts are consumed evidence, not reusable test sets.
5. Run `uv sync --extra dev` and `uv run pytest`.

## Research contribution requirements

An experiment contribution should include:

- a falsifiable hypothesis;
- data splits and seed roles fixed before evaluation;
- metrics, thresholds, negative controls, and stopping rules;
- the exact command needed to reproduce the run;
- machine-readable results with provenance;
- a report that includes limitations and negative findings.

New blind evaluations must use a new, externally held split or an equivalent
mechanism that prevents contributors from inspecting the evaluation episodes
during development.

## Code style

- Use deterministic seeds and configuration files instead of inline
  experiment changes.
- Add tests for behavior changes.
- Keep learner inputs separate from evaluation-only truth.
- Match the language already used by the file.
- Preserve historical `calmodel` paths in frozen artifacts.

By submitting a contribution, you agree that it is licensed under the
repository's applicable Apache-2.0 or CC BY 4.0 license.
