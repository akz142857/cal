# Cal

Cal explores whether an embodied learner can discover a stable
self–environment boundary from unlabeled sensorimotor experience.

The complete milestones, hypotheses, controls, and acceptance criteria are in
[the research plan](docs/RESEARCH_PLAN.md).
The Python namespace is `cal`; the historical rename and frozen-protocol
handling are documented in [the package migration note](docs/PACKAGE_RENAME.md).

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
cal/
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
python -m cal.learning.replay artifacts/trajectory-000.json \
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
python -m cal.learning.replay artifacts/trajectory-000.jsonl.gz \
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
uv run cal-train experiments/baseline.yaml \
  --output results/M1-gru-full-seed000
```

The runner records the learned model's validation history together with the
copy-last-observation and mean-observation baselines. Model inputs never
include evaluation masks.

The first measured baseline and its limitations are documented in
[the M1 prediction report](docs/experiments/M1_PREDICTION_BASELINE.md).

Evaluate whether the frozen state linearly exposes the body mask:

```bash
uv run cal-probe \
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
uv run cal-adapt \
  results/M1-gru-full-seed000/checkpoint.pt \
  experiments/baseline.yaml \
  --output results/M1-body-adaptation-seed000
```

The first adaptation curves and their failure cases are documented in
[the M1 adaptation report](docs/experiments/M1_ADAPTATION.md).

Probe whether frozen state distinguishes self-commanded change from additional
exogenous object motion:

```bash
uv run cal-cause-probe \
  results/M1-gru-full-seed000/checkpoint.pt \
  experiments/baseline.yaml \
  --output results/M1-cause-full-seed000
```

The first causal-readout result, including the chance-level negative finding,
is documented in
[the M1 cause-probe report](docs/experiments/M1_CAUSE_PROBE.md).

Run the recoverable five-seed M1 suite and rebuild its aggregate:

```bash
uv run cal-multiseed --output results/M1-multiseed
uv run cal-m1-summary --output results/M1-stage-summary.json
uv run cal-index --results results
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
retention is 0.9792, and M3 true-graph probability is 0.999991. The
simulator visibility mask has since been removed from M4: the unprivileged
variant infers occlusion by shadow-casting over its own sensed occupancy,
maintains probabilistic motion hypotheses with online pause-regularity
learning for occluded objects, and passes its one-shot frozen holdout
(occupancy IoU 0.7504, moving-hidden probability 0.7492, with the
assume-all-visible control failing as required). The chain now authorizes a
reconnection design review toward the original object-permanence M2; see
[the unprivileged M4 report](docs/experiments/V2_M4_UNPRIVILEGED_REPORT.md).
Rebuild development artifacts with:

```bash
uv run cal-v2-identifiability
uv run cal-v2-diagnostic-ceiling
uv run cal-v2-causal-sufficiency
uv run cal-v2-audit-summary
uv run cal-v2-m1
uv run cal-v2-m2
uv run cal-v2-m2-review
uv run cal-v2-m3-hypotheses --split development
uv run cal-v2-m3-review
uv run cal-v2-m1-m3-confirm --split development
uv run cal-v2-m1-m3-confirm-review
uv run cal-v2-m4 --exploratory
uv run cal-v2-m4-unprivileged --split development
uv run cal-v2-stage-summary
uv run cal-index --results results
```

Launch the read-only project dashboard with:

```bash
uv run streamlit run streamlit_app.py
```

The dashboard reads persisted JSON artifacts only. It shows the V2 stage
chain, fresh M1–M3 confirmation, mechanism controls, protocol audit, resource
budget, and per-seed confirmation episodes; it does not rerun experiments or
rewrite frozen protocols.

The next I1 architecture, a unified entity belief graph, subsequently passed
its reviewed calibration, one-shot validation, and one-shot holdout with all
13 gates true. Generate its presentation-only interactive replay with:

```bash
uv run cal-v2-i1-replay --seed 30000
uv run cal-v2-i1-replay --seed 30000 \
  --check docs/experiments/assets/v2_i1_v4_replay_seed30000.html
```

The replay is restricted to repeatable calibration seeds and never consumes
validation or holdout seeds. See the
[I1 final report](docs/experiments/V2_I1_NEXT_ARCHITECTURE_RESULT.md) and
[replay guide](docs/experiments/V2_I1_REPLAY_GUIDE.md).

The next L0 probe freezes I1 and tests whether controlled Chinese propositions
about self, spatial relations, identity after reappearance, and hidden-object
permanence are linearly readable from its entity state. Its shortcut-resistant
V4 development run passes every frozen gate and three final independent
reviews. Its V5 exact-source-lock and one-shot origin registry are implemented;
the only V5 review-holdout attempt was authorized and consumed, but stopped
before producing metrics because its frozen identity-scramble control could not
be constructed from the holdout events. V5 therefore has no passing holdout
claim and cannot be retried. A preregistered V6 row-local counterfactual removes
that structural dependency and passes all 24 development gates on a clean
implementation commit. After review hardening, three independent final reviews
report no P0/P1/P2 findings. Its V7 exact-source lock was published as tag
object `b8b391abc5b54aa7acbf58bef6a6cdf2c7d32664`, targeting
`db524a3d1b65a232c2159541a79d7098227848f5`. Post-lock independent review found
one-shot crash-recovery defects, so V7 remains unopened and unauthorized and
must never be consumed. Those defects are being repaired in preparation for a
new immutable V8 protocol and tag namespace. See the
[L0 language-readout report](docs/experiments/V2_L0_LANGUAGE_READOUT.md).

The measured evidence, limitations, and M4 decision are documented in the
[V2 audit report](docs/experiments/V2_AUDIT_REPORT.md) and
[V2 stage report](docs/experiments/V2_STAGE_REPORT.md). The one-shot reviews
are documented separately in the
[probabilistic association report](docs/experiments/V2_M2_PROBABILISTIC_ASSOCIATION_REPORT.md)
and the
[complete-body-graph report](docs/experiments/V2_M3_BODY_GRAPH_HYPOTHESIS_REPORT.md),
and the
[fresh M1–M3 confirmation report](docs/experiments/V2_M1_M3_INTEGRATED_CONFIRMATION_REPORT.md).
