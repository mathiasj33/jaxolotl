# jaxolotl Agent Instructions

- Run Python through pixi in this workspace. Use `pixi run python ...` for scripts, `pixi run pytest --ignore=runs` for tests, and `pixi run ruff check ...` / `pixi run ruff format ...` for linting and formatting.
- The project targets Python 3.12 only; keep changes compatible with the version pinned in [pyproject.toml](pyproject.toml).
- Treat [src/jaxolotl/](src/jaxolotl) as the main library code and [scripts/](scripts) as runnable utilities. One-off scripts should go in [local/](local/).
- Tests are generally co-located to their associated implementation files, with a `_test` suffix.
- Keep tests reasonable: do write isolated unit tests for core logic and larger integration tests, but only if appropriate and useful. Do not write integration tests that mock large parts of the codebase and merely assert implementation-specific details (such as functions being called). If in doubt, prefer to not write a test rather than writing a test.
- Do not by default run the entire test suite after every change. Do run targeted tests where appropriate.
- When working on long-horizon tasks (more than simple refactorings, e.g. complex changes or multiple commits) always keep a concise log of your progress and design decisions in a Markdown file under `.artifacts`. This log should have a semantic name of the task and include a timestep in the header of the file.