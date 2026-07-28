# Reproducibility

## Environment

- Python 3.11 or newer;
- `uv`;
- CPU execution is sufficient for the published V2 mechanism experiments;
- dependency versions are locked in `uv.lock`.

Set up the project and run the complete test suite:

```bash
uv sync --extra dev
uv run pytest
```

## Ten-minute verification

Verify the checked-in result index and launch the read-only dashboard:

```bash
uv run cal-index --results results
uv run streamlit run streamlit_app.py
```

Generate a presentation-only replay from a permitted development seed:

```bash
uv run cal-v2-i1-replay --seed 30000
uv run cal-v2-l0-language-replay --seed 33100
```

The generated replays reject validation or holdout seeds that are not approved
for presentation.

The checked-in replay HTML files are immutable presentation references with
fixed SHA-256 digests. Regeneration is deterministic on the same runtime and
CPU architecture. Across architectures, numerically tied identity hypotheses
can be ordered differently, so individual presentation frames are not claimed
to be byte-identical. Cross-platform verification therefore checks the frozen
artifact digest plus the protocol, action schedule, evidence boundary, and
formal aggregate metrics. The replay remains presentation-only and is not a
new evidence artifact.

## Rebuilding published development results

The command sequence for each stage is maintained in `README.md`. Every
machine-readable result records:

- the Git commit and dirty state;
- combined and per-file source hashes;
- configuration and seeds;
- Python, dependency, and operating-system versions;
- metrics, gates, and decision.

Use:

```bash
uv run cal-index --results results
```

to rebuild `results/INDEX.json`.

## One-shot evidence

Do not rerun historical holdout commands and describe them as independent
evidence. The V8 L0 holdout has already been consumed. Its exact result bytes
are checked in and cryptographically bound to
`calmodel-l0-v8-holdout-terminal-evidence`.

To verify the binding:

```bash
git cat-file -p calmodel-l0-v8-holdout-terminal-evidence
shasum -a 256 results/V2-L0-language-readout-holdout-v8.json
```

The recorded result SHA-256 must be:

```text
ae5ad9d4ef457d22680dc30048bf8e0421f5e708c724351b8751f8061a2d9d04
```

Independent replication should create a new protocol and externally held
evaluation split instead of selecting new seeds after inspecting outcomes.
