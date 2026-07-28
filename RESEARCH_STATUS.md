# Research status

Last updated: 2026-07-28.

Cal is an open research prototype. It studies whether an embodied learner can
form an explicit, persistent entity state from action-conditioned sensory
experience and whether controlled language propositions are readable from that
state.

## Evidence summary

| Stage | Status | Evidence boundary |
| --- | --- | --- |
| V1 M1 | Did not pass | Predictive representation did not satisfy the original frozen body-discovery gate. |
| V1 M1b–M1v | Completed negative program | None of the preregistered mechanism candidates passed the frozen envelope gate. |
| V2 audits and M1 | Passed | Identifiability, diagnostic-ceiling, causal-sufficiency, and self-identification gates passed in the simulator. |
| V2 M2 | Passed after a documented failure and redesign | Probabilistic multi-trajectory association passed its frozen holdout. |
| V2 M3 | Passed | The complete-body-graph posterior preserved ambiguity and resolved it after symmetry breaking. |
| V2 M4 unprivileged | Passed | Hidden occupancy and motion hypotheses passed without access to the simulator visibility mask. |
| V2 I1 entity belief graph | Passed | Calibration, validation, and one-shot holdout passed all 13 gates. |
| V2 L0 language readout V8 | **Did not pass** | The unique one-shot holdout was consumed on 2026-07-28 and ended with `stop_and_report`: 21 of 24 gates passed and 3 failed. |

## V8 L0 holdout result

The immutable result is
`results/V2-L0-language-readout-holdout-v8.json`, SHA-256
`ae5ad9d4ef457d22680dc30048bf8e0421f5e708c724351b8751f8061a2d9d04`.
It was produced from clean commit
`e26c613e4648528f38f7125b662c6daf89448983` and is bound to the Git tag
`calmodel-l0-v8-holdout-terminal-evidence`.

The formal entity graph reached:

| Metric | Balanced accuracy |
| --- | ---: |
| Macro | 0.8903 |
| Self | 0.9993 |
| Spatial | 0.8750 |
| Identity after reappearance | 0.8847 |
| Hidden-object permanence | 0.8021 |

The run nevertheless failed three frozen controls:

- `formal_beats_raw_permanence`: raw sensor permanence was 0.8750;
- `all_visible_permanence_fails`: the assume-all-visible control reached
  0.8542;
- `identity_scramble_integrity_pass`: the preregistered integrity condition
  was not met because opposite-motion coverage was 0.8777 rather than
  complete.

This result supports a narrower statement: much of the controlled self,
spatial, and identity information is linearly readable from the frozen entity
state in this synthetic environment. It does **not** independently verify the
full L0 language-readability claim, and it does not establish that
hidden-object permanence depends on the formal entity graph.

## Claims this repository does not make

Cal is not currently:

- a general world model;
- validated on RGB video, physical robots, or real-world datasets;
- a production or safety-critical system;
- evidence of open-ended language understanding;
- proof that the learned mechanisms generalize beyond the documented
  simulator and seed distributions.

Historical holdouts are consumed evidence and must not be reused as fresh
evaluation sets. New claims require a newly preregistered split whose contents
remain outside the development process.
