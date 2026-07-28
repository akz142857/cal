# Cal data card

## Summary

The checked-in data consists of synthetic experiment summaries, protocol
records, and project-created demonstrations. Sensorimotor trajectories are
generated deterministically by Cal's 2D simulator; the repository does not
contain human-subject data or scraped personal data.

## Modalities

Depending on the experiment, generated observations include:

- low-resolution binary vision or sensed occupancy;
- proprioceptive joint state;
- binary touch;
- discrete actions;
- deterministic next observations;
- evaluation-only masks, entity identities, or simulator truth used only by
  probes and metrics.

Evaluation-only state is not part of learner input. Runtime audits for the
source-locked V2 experiments check this boundary.

## Storage

- `experiments/`: YAML configurations and frozen JSON protocols;
- `results/`: selected JSON summaries and the result index;
- `docs/experiments/`: permanent human-readable reports;
- `docs/experiments/assets/`: project-created HTML/GIF replays;
- `artifacts/` and `checkpoints/`: locally generated, ignored by Git.

Large trajectories and checkpoints are intentionally excluded from Git. They
can be regenerated from the corresponding commit, configuration, and seeds.
If a future release publishes them separately, it must include checksums,
schema, provenance, and the producing source commit.

## Splits and leakage

Seeds have explicit roles such as development, validation, confirmation, and
holdout. Historical holdout seeds and outputs are now public consumed
evidence. They must never be treated as blind evaluation data again.

Future independent evaluations must use an externally held split, evaluation
service, or independent custodian. Publishing only a numeric seed is
insufficient when the simulator and generation code are public.

## Limitations

The current environment is small, deterministic, two-dimensional, and
synthetic. It lacks natural RGB appearance, sensor noise representative of
physical systems, complex backgrounds, human behavior, and real-world domain
shift. Results should not be interpreted as real-world robot performance.

## License and citation

Project-created result summaries, reports, and demonstrations are licensed
under CC BY 4.0 as described in `LICENSE-DATA.md`. Source code and
configurations are Apache-2.0. Cite the exact release using `CITATION.cff`.
