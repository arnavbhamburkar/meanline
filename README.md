# meanline

Mean-line (1D, meridional mean radius) design and analysis of axial compressors
in Python. Work in progress: single stage first, then multi-stage.

This is a mean-line model at one radius, with annulus and blockage assumptions
stated explicitly. It is not a 3D result.

See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the plan and validation strategy.

## Development

```
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
