# jaxolotl Agent Instructions

- Run Python through pixi in this workspace. Use `pixi run python ...` for scripts, `pixi run pytest` for tests, and `pixi run ruff check ...` / `pixi run ruff format ...` for linting and formatting.
- The project targets Python 3.12 only; keep changes compatible with the version pinned in [pyproject.toml](pyproject.toml).
- Treat [src/jaxolotl/](src/jaxolotl) as the main library code and [scripts/](scripts) as runnable utilities.
- Tests are generally co-located to their associated implementation files, with a `_test` suffix.