# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Cal is a research codebase investigating whether an embodied learner can
discover a stable self–environment boundary from unlabeled sensorimotor
experience, in a deterministic 2D simulated world. It is organized as a
sequence of preregistered, gated experiments (M1, M1b–M1v, then a V2 program:
identifiability/diagnostic-ceiling/causal-sufficiency audits, M1–M4), not as a
product codebase. Changes to evaluation logic have scientific-integrity
implications — see "Frozen protocols" below before touching anything under
`cal/evaluation/` or `experiments/*.json`.

Full context lives in `docs/RESEARCH_PLAN.md` (V1, M1–M1v) and
`docs/RESEARCH_PLAN_V2.md` (V2, current program) — both written in Chinese.
`README.md` is the up-to-date narrative of what has been run and what passed
or failed; read it before assuming what stage the project is in.

## Setup and commands

```bash
uv sync --extra dev        # required before running anything, including tests
uv run pytest              # run the full test suite
uv run pytest tests/test_world.py               # single test file
uv run pytest tests/test_world.py::test_name -q  # single test
```

There is no separate lint command configured; pytest is the only checked
command (`[tool.pytest.ini_options]` in `pyproject.toml`, `testpaths = ["tests"]`).

All experiment/eval entry points are `console_scripts` declared in
`pyproject.toml` and invoked via `uv run <name>`, e.g.:

```bash
uv run cal-train experiments/baseline.yaml --output results/M1-gru-full-seed000
uv run cal-probe results/.../checkpoint.pt experiments/baseline.yaml --output results/...
uv run cal-multiseed --output results/M1-multiseed
uv run cal-index --results results        # rebuilds results/INDEX.json
uv run streamlit run streamlit_app.py     # read-only dashboard over results/ JSON
```

See `README.md` for the full, current command sequence for both the V1 (M1)
and V2 (identifiability → M1 → M2 → M3 → M4) pipelines — it is kept in sync
with what has actually been run and is more reliable than inferring commands
from `pyproject.toml` alone.

## Package naming

The Python package is `cal` (source dir `cal/`, imports `cal.*`, script
prefix `cal-*`). It was renamed from `calmodel`; see `docs/PACKAGE_RENAME.md`.
Frozen protocol JSON files and historical one-shot confirmation results still
reference the old `calmodel/...` paths on purpose — those are historical
evidence and must never be rewritten to `cal/...`. Do not "fix" old
`calmodel` paths you encounter inside `experiments/*.json`,
`experiments/*.sha256`, or already-published `docs/experiments/*.md`.
`tests/test_package_layout.py` explicitly asserts `calmodel` is NOT
importable.

## Architecture

```
cal/
  env/          deterministic 2D world, articulated body, sensors (vision/proprioception/touch)
  model/        modality encoders, recurrent core (GRU), predictors, entity graph, body hypotheses
  learning/     experience replay/recording, sequence datasets, training loop
  evaluation/   probes, ablations, and the staged M1/M1b.../V2 experiment scripts
  infra/        provenance capture (source hashing, git state) and results indexing
experiments/    YAML configs for trainable runs; frozen JSON protocols + .sha256 sidecars for V2 confirmation stages
results/        run outputs (JSON summaries + INDEX.json); not meant to be hand-edited
docs/experiments/  the permanent written record of each stage's numbers and decisions
```

Data flow for a basic run: `cal.env.world` generates deterministic
trajectories → `cal.learning.replay` records/replays them as transitions
(`observation(t) + action(t) -> observation(t+1)`), storable as JSON or a
versioned `.jsonl.gz` compressed stream → `cal.learning.trainer` trains a
`cal.model.recurrent_core` (GRU) predictor over encoded multimodal
observations → `cal.evaluation.*` scripts freeze the trained model and probe
it (linear body probe, cause probe, adaptation, counterfactual, etc.) without
ever letting probe gradients touch the prediction model. Ground-truth body
masks and other evaluation-only state are generated only during evaluation by
deterministic world replay — they are never part of model inputs.

### Frozen protocols and source locks (important, and unevenly enforced)

The V2 confirmation stage (`cal/evaluation/v2_m1_m3_confirmation.py`) is
gated by a preregistered protocol JSON in `experiments/` (currently
`V2_M1_M3_INTEGRATED_CONFIRMATION_PROTOCOL_V3.json`; V1/V2 are superseded but
kept as historical amendment-chain links — see their `amendment_record`),
protected by a sibling
`.sha256` file **and** by a `locked_source_sha256` map embedded in the
protocol that pins hashes of `cal/evaluation/v2_m1.py`, `v2_m2.py`,
`v2_m3_hypotheses.py`, `cal/model/online_control.py`, `entity_graph.py`, and
`body_hypotheses.py`. Only this confirmation script actually calls
`_verify_locked_sources` and hashes those dependency files at runtime — **if
you edit any of the six files above, `v2_m1_m3_confirmation.py` runs will
raise on the next execution**, by design.

This enforcement is *not* uniform across the pipeline, and treating it as if
it were will give a false sense of safety:
- `v2_m2.py` and `v2_m3_hypotheses.py` each hash-check only their own
  protocol JSON (`_load_frozen_protocol`) — they do not verify
  `entity_graph.py`/`body_hypotheses.py` against any locked hash themselves.
  Running `cal-v2-m2` or `cal-v2-m3-hypotheses` standalone will not detect
  that a dependency changed since freeze; only a later
  `cal-v2-m1-m3-confirm` run would catch it.
- `v2_m2.py`'s `--protocol`/`--split` flags are optional; omitting them
  silently runs an unlocked `"legacy_development"` path with no hash
  verification at all.
- `v2_m1.py` and `v2_m4.py` have no protocol JSON or source lock whatsoever —
  their thresholds are inline literals, checked only by
  `require_authorization` on the *previous* stage's output, not by any
  hash-locked source.
- `locked_source_sha256` in the confirmation protocol does not cover
  `cal/evaluation/v2_m3.py` (which `v2_m3_hypotheses.py` imports `_arm`/
  `_rasterize` from) or `cal/env/world.py`/`point_world.py` (the ground-truth
  simulators formal agents are scored against) — changes to those files after
  freeze go undetected everywhere.
- `capture_provenance`'s broad `source_sha256` (below) is descriptive only;
  nothing validates a result's provenance hash against a locked value, so it
  does not itself enforce immutability.

If you need to change one of the six locked files, that requires a new
protocol version/amendment (see existing `*_V2.json` / `amendment_record`
patterns), not an in-place edit. If you touch `v2_m2.py`, `v2_m3_hypotheses.py`,
`v2_m1.py`, `v2_m3.py`, or the env simulators, be aware no automated check
will flag it except a subsequent confirmation-stage run (and even that only
covers a subset of these files) — don't assume silence means safe.

Result JSON produced by these stages is validated with
`cal.evaluation.v2_artifacts.require_authorization`, which enforces
`result_schema_version == 1`, a matching `decision`, complete `gates`, and
present `provenance` before a downstream stage is allowed to treat an
upstream result as passing.

### Provenance

`cal.infra.provenance.capture_provenance` stamps every result JSON with a
combined source SHA-256 (derived from all tracked source files' names and
contents — unique even without a git commit), per-file hashes, git commit/
dirty state, and dependency/OS versions. This is how `docs/RESULT_FORMAT.md`
results stay reproducible/attributable without committing `results/` itself
(which is generally not version-controlled; only the durable numbers and
conclusions written into `docs/experiments/*.md` are).

## Conventions worth knowing

- Config-driven: experiment behavior (image size, modalities enabled, seeds,
  hyperparameters) is described by YAML files in `experiments/`, not by
  editing Python. `shuffle_modalities`, `include_action`, `include_touch`,
  etc. toggle ablations from the same trainer/eval code path.
- Seeds are partitioned by role (train/validation/test/probe/adaptation/
  cause_probe) directly in the experiment YAML, not derived from a single
  global seed.
- Dataclasses (`@dataclass(frozen=True, slots=True)`) are the standard shape
  for typed config/result structures throughout `cal/`.
- Docstrings and comments in this repo are in English; the two research-plan
  documents and a few evaluation-support docs are in Chinese — match the
  existing language of a file you're editing rather than converting it.
