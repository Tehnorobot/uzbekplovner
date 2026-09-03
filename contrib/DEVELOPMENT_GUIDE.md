<!-- TODO -->

# Format & linting test

Setup env
```bash
uv sync --extra dev
```

To check
```bash
uv run ruff format --check
ruff check
```

To fix
```bash
uv run ruff format
ruff check --fix
```


# e2e

```bash
bash e2e_test.sh
```
