# Core compatibility area

The authoritative POC service implementation lives in the repository-root
`services/` directory. This directory is not a second service source tree.

The overlapping Python modules retained here are a compatibility mirror for
historical import paths. Update `services/` first, then copy the exact file
into `core/`. The CI workflow runs
`scripts/check-core-service-sync.py` and fails if shared non-test modules
drift. `conftest.py` files are independent because they configure different
test roots.

The active repository-local reviewer tooling remains here:

- `claude-code-skills/`
- `reviewer-mcp/`
- `tools/` and related integration scripts

The retired Alert and Messenger service implementations were moved to
`archive/legacy-core/`; their Messenger-only tests were moved to
`archive/legacy-core-tests/`. They are not started, installed, or used by the
POC runtime.

For service inventory, startup, API contracts, and test instructions, use the
[root README](../README.md).
