# Contributing

Focused corrections and lesson improvements are welcome. Please open an issue before proposing a
new model or dataset so the lab stays small and sequential.

## Local checks

```bash
python -m pip install -e ".[notebooks,dev]"
make check
```

When changing a lesson, edit `scripts/build_notebooks.py`, run `make notebooks`, and commit the
generated notebook. Committed notebooks must have no execution counts or outputs.

## Pull requests

- Explain the behavior being changed and why.
- Add or update tests for shared code.
- Report commands run and their results.
- Do not commit datasets, model checkpoints, credentials, or notebook output.
- Keep performance claims tied to a committed configuration and inspectable artifact.
