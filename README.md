# Cal

Cal explores whether an embodied learner can discover a stable
self–environment boundary from unlabeled sensorimotor experience.

The complete milestones, hypotheses, controls, and acceptance criteria are in
[the research plan](docs/RESEARCH_PLAN.md).

## First experiment: body discovery

The initial agent observes:

- a low-resolution visual frame;
- proprioceptive joint state;
- binary touch signals;
- the action it just executed.

The learner is trained to predict the next observation. Body labels are never
used during representation learning; they are available only to evaluation
probes.

The first comparison contains four conditions:

| Experiment | Available signals |
| --- | --- |
| `baseline` | vision, proprioception, touch, action |
| `no_action` | vision, proprioception, touch |
| `no_touch` | vision, proprioception, action |
| `shuffled_modalities` | all signals with broken temporal alignment |

## Repository layout

```text
calmodel/
  env/          2D world, body, and sensors
  model/        modality encoders, recurrent state, and predictors
  learning/     training loop and experience replay
  evaluation/   body probes and experiment metrics
experiments/    reproducible experiment configurations
tests/          unit and integration tests
```

## Development order

1. Implement deterministic body dynamics and sensors.
2. Generate and replay random-action trajectories.
3. Train the recurrent multimodal predictor.
4. Freeze it and evaluate body information with a linear probe.
5. Run the three ablation experiments.

## Record a trajectory

Generate 100 learner-facing transitions:

```bash
python -m calmodel.learning.replay artifacts/trajectory-000.json \
  --steps 100 \
  --seed 0
```

Each transition has the form:

```text
observation(t) + action(t) -> observation(t+1)
```

The JSON file contains vision, proprioception, touch, actions, and the
configuration required for deterministic replay. It deliberately excludes
body masks, object masks, and all other evaluation-only state.

Use a `.jsonl.gz` suffix for the versioned compressed stream format:

```bash
python -m calmodel.learning.replay artifacts/trajectory-000.jsonl.gz \
  --steps 100 \
  --seed 0
```

`iter_compressed_experiences` validates and yields one transition at a time,
`verify_compressed_trajectory` strictly replays without materializing the
trajectory, and `CompressedTrajectorySequenceIterableDataset` buffers only one
training window.

## Train the first prediction baseline

Create the project environment and run the GRU baseline:

```bash
uv sync --extra dev
uv run calmodel-train experiments/baseline.yaml \
  --output results/M1-gru-full-seed000
```

The runner records the learned model's validation history together with the
copy-last-observation and mean-observation baselines. Model inputs never
include evaluation masks.

The first measured baseline and its limitations are documented in
[the M1 prediction report](docs/experiments/M1_PREDICTION_BASELINE.md).

Evaluate whether the frozen state linearly exposes the body mask:

```bash
uv run calmodel-probe \
  results/M1-gru-full-seed000/checkpoint.pt \
  experiments/baseline.yaml \
  --output results/M1-gru-body-probe-seed000
```

The body masks are regenerated only during evaluation by deterministic world
replay. Probe gradients cannot reach the prediction model.

The first frozen-probe results and their limitations are documented in
[the M1 body-probe report](docs/experiments/M1_BODY_PROBE.md).

The first modality and temporal controls are documented in
[the M1 ablation report](docs/experiments/M1_ABLATIONS.md).

Evaluate zero-shot prediction and finite-experience adaptation after body or
sensor changes:

```bash
uv run calmodel-adapt \
  results/M1-gru-full-seed000/checkpoint.pt \
  experiments/baseline.yaml \
  --output results/M1-body-adaptation-seed000
```

The first adaptation curves and their failure cases are documented in
[the M1 adaptation report](docs/experiments/M1_ADAPTATION.md).

Probe whether frozen state distinguishes self-commanded change from additional
exogenous object motion:

```bash
uv run calmodel-cause-probe \
  results/M1-gru-full-seed000/checkpoint.pt \
  experiments/baseline.yaml \
  --output results/M1-cause-full-seed000
```

The first causal-readout result, including the chance-level negative finding,
is documented in
[the M1 cause-probe report](docs/experiments/M1_CAUSE_PROBE.md).

Run the recoverable five-seed M1 suite and rebuild its aggregate:

```bash
uv run calmodel-multiseed --output results/M1-multiseed
uv run calmodel-m1-summary --output results/M1-stage-summary.json
uv run calmodel-index --results results
```

The complete acceptance decision is in
[the M1 stage report](docs/experiments/M1_STAGE_REPORT.md). M1 did not pass:
the project has therefore returned to the M1b action-causality mechanism
instead of proceeding to M2. The versioned result/provenance contract is
documented in [the result format](docs/RESULT_FORMAT.md).

The first M1b mechanism screen and its remaining limitations are documented in
[the M1b stage report](docs/experiments/M1B_STAGE_REPORT.md).

The subsequent preregistered M1c–M1v mechanism program is complete. None of
the direct-envelope, spatial-state, global-query, curriculum, competitive-slot,
or complete action-basis candidates passed the frozen envelope gate. The
evidence and stopping decision are in the
[M1 extended stage report](docs/experiments/M1_EXTENDED_STAGE_REPORT.md).
M2 remains intentionally unstarted.

The next research hypothesis is defined in the
[Cal V2 research plan](docs/RESEARCH_PLAN_V2.md). V2 is vision-first,
failure-driven and resource-bounded: it begins with identifiability,
supervised-information-ceiling and causal-sufficiency audits, then permits an
online controllable-entity learner only if those audits pass.

V2-A–C and M1 pass their corrected gates. The reviewed M2 failure
(0.8125 crossing identity retention) was addressed with motion/geometry-aware
probabilistic multi-trajectory association. Its implementation-first frozen
holdout passes with 1.000 identity retention and zero switches. M3 now uses a
normalized categorical posterior over complete body graphs. Its one-shot
frozen holdout preserves the observable 0.5/0.5 symmetry, then reaches
0.999994 mean true-graph probability within at most two steps after symmetry
break. A separate one-shot fresh-data confirmation then revalidates M1–M3
without reading either old holdout: M1 F1 is 0.9995, M2 crossing identity
retention is 0.9792, and M3 true-graph probability is 0.999991. M4 still uses
a privileged simulator visibility mask, so the formal chain now stops at M4.
Rebuild development artifacts with:

```bash
uv run calmodel-v2-identifiability
uv run calmodel-v2-diagnostic-ceiling
uv run calmodel-v2-causal-sufficiency
uv run calmodel-v2-audit-summary
uv run calmodel-v2-m1
uv run calmodel-v2-m2
uv run calmodel-v2-m2-review
uv run calmodel-v2-m3-hypotheses --split development
uv run calmodel-v2-m3-review
uv run calmodel-v2-m1-m3-confirm --split development
uv run calmodel-v2-m1-m3-confirm-review
uv run calmodel-v2-m4 --exploratory
uv run calmodel-v2-stage-summary
uv run calmodel-index --results results
```

Launch the read-only project dashboard with:

```bash
uv run streamlit run streamlit_app.py
```

The dashboard reads persisted JSON artifacts only. It shows the V2 stage
chain, fresh M1–M3 confirmation, mechanism controls, protocol audit, resource
budget, and per-seed confirmation episodes; it does not rerun experiments or
rewrite frozen protocols.

The measured evidence, limitations, and decision to stop at M4's visual-only gate are documented in the
[V2 audit report](docs/experiments/V2_AUDIT_REPORT.md) and
[V2 stage report](docs/experiments/V2_STAGE_REPORT.md). The one-shot reviews
are documented separately in the
[probabilistic association report](docs/experiments/V2_M2_PROBABILISTIC_ASSOCIATION_REPORT.md)
and the
[complete-body-graph report](docs/experiments/V2_M3_BODY_GRAPH_HYPOTHESIS_REPORT.md),
and the
[fresh M1–M3 confirmation report](docs/experiments/V2_M1_M3_INTEGRATED_CONFIRMATION_REPORT.md).
