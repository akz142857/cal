# Open-source release audit

Audit date: 2026-07-28.

## Outcome

The repository is suitable for an open research preview subject to the
scientific claim boundaries in `RESEARCH_STATUS.md`.

## Secret and sensitive-file checks

- Gitleaks scanned all 45 reachable commits, approximately 7.08 MB of Git
  history, with redaction enabled: no leaks found.
- Gitleaks separately scanned the exact 6.41 MB release file set, including
  untracked files scheduled for the release and excluding Git-ignored
  environments: no leaks found.
- A second pattern scan checked the working tree and complete patch history
  for common cloud keys, GitHub tokens, OpenAI-style keys, Slack tokens,
  credential-bearing URLs, and private-key headers: no matches found.
- Tracked filenames were checked for environment files, private keys,
  credentials, tokens, and keystores: no matches found.
- `.venv`, test caches, generated artifacts, checkpoints, and non-selected
  results are ignored.
- No Git blob larger than 5 MB exists in reachable history.

These checks reduce risk but cannot prove that no sensitive information
exists. Contributors must continue to review changes before pushing.

## Dependency check

`pip-audit` checked the installed dependency set corresponding to the locked
project environment and reported no known vulnerabilities on 2026-07-28.
The local package `cal` is not a PyPI dependency and was correctly skipped.

## Identity and privacy

Reachable commits contain two historical author addresses:

- `me.zyliu@gmail.com`;
- `ziyang.liu@wahool.com`.

Rewriting them would change every affected commit ID and invalidate published
source-lock and evidence tags. The history was therefore preserved.
`.mailmap` maps both addresses to
`209027+akz142857@users.noreply.github.com` for standard Git author display,
but the original addresses remain readable in raw historical objects. This is
an accepted residual disclosure for preservation of the research evidence
chain.

## GitHub repository settings

- visibility: public;
- Discussions: enabled;
- private vulnerability reporting: enabled;
- GitHub Secret Protection and push protection: enabled;
- Issues and pull requests: enabled.

## Licensing

- source code, tests, and experiment configurations: Apache-2.0;
- project-created documentation, selected result summaries, and demonstration
  media: CC BY 4.0;
- cited third-party works are not relicensed.

No repository-owned patent search or legal opinion was performed. Anyone with
unpublished patent claims or third-party contractual obligations should
resolve them before contributing affected material.

## Scientific-integrity checks

- the V8 result bytes match the terminal-evidence SHA-256;
- the result records a clean frozen commit at run start and end;
- the unique holdout is documented as consumed and failed;
- no passing claim is made for L0 V8;
- future blind tests require a new externally held split.
