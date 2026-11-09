# POC service source layout

The canonical REST service implementations and tests are under this directory.
Install their pinned Python runtime dependencies from the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest services -q
```

The `core/` directory is a compatibility mirror and repository-local
reviewer tooling area. Implement service changes under `services/` first.
