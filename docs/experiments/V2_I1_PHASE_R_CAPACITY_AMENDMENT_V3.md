# I1-P1 Phase R capacity amendment v3

> **Status: frozen development-only Phase R amendment.**
>
> This amendment changes only the candidate-side packed capacity prototype and
> its privileged conformance diagnostic. It does not change the Phase 0
> statistical design, its V10 artifact, any evaluation gate, or any
> validation/holdout stream. It authorizes no validation or holdout use.

## Motivation and prior evidence

The model-blind Phase 0 source expansion from 16 to 64 development seeds also
expanded the privileged Phase R conformance population from 221 to 741
occlusion episodes. The locked K=48 rerun produced a valid development
`phase_r_no_go`: maximum cumulative pruned mass was
`0.9995860677181558` and maximum position TV was `0.20471206293731348`.

Candidate-independent development capacity scans then gave:

| K | maximum cumulative pruned mass | maximum position TV | declared active bytes |
|---:|---:|---:|---:|
| 48 | 0.9995861 | 0.2047121 | 52,145 |
| 64 | 0.9474372 | 0.0885593 | 64,625 |
| 80 | 0.0322553 | 0.0223774 | 77,105 |
| 96 | 4.33e-15 | 6.10e-8 | 89,585 |

K=96 is the first scanned capacity that passes both unchanged conformance
limits (`<=0.01`). The scan is a privileged development feasibility study, not
confirmatory model evidence and not eligible for candidate selection.

## Frozen v3 capacity construction

- `H_max=5`, `E_max=11`, `K_max=96`, and `S_max=5*11*96=5,280`.
- The complete detached posterior bank remains flat `uint16` state codes plus
  `float32` probabilities. No copy-on-write sharing is assumed.
- Expansion retains the shared `12*K` code/probability workspace and the direct
  state-index accumulator.
- Atomic factor replacement validates code range, shape, finiteness,
  non-negativity, normalization, and capacity before touching live state.
- After validation, replacement stages exactly one K-sized factor and commits
  its aligned live slices. Whole-bank scratch copies are forbidden: they do not
  improve rollback because no fallible operation remains after validation.
- Overflow and every validation failure must leave codes, probabilities, and
  counts byte-for-byte unchanged.
- The unchanged limits are 65,536 active bytes, 100,000 parameters, and
  5,000,000 estimated MAC/step. Both declared array accounting plus the 8,192 B
  Python guard and measured deep size must pass.

The formal Phase R v3 runner must use the complete 64-seed development registry,
`turn_p=0.35`, and all 741 collected conformance episodes. It must regenerate a
new canonical artifact and source lock. Any conformance, resource, kernel,
branch-accounting, workspace, or atomicity gate failure remains a fail-closed
No-Go. The earlier Phase R v2 artifact remains immutable No-Go evidence.
