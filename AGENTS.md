# AGENTS.md

Repository instructions for agents working on FaceFusion Ray Serve API.

## Scope

- Project-owned code lives in `config.py`, `main_serve.py`, `facefusion_job.py`,
  `models.py`, tests, and repository documentation.
- `facefusion/`, `facefusion.py`, and `install.py` are vendored upstream
  FaceFusion code. Do not edit, format, lint, or type-check them unless the task
  is explicitly to refresh the vendor snapshot.
- Keep generated text, comments, commit messages, and documentation in standard
  American English.

## Environment

The project uses uv for dependency management and virtualenv creation.

Install all dependencies:

```bash
uv sync
```

Python 3.13 is managed by uv. Run `uv python install 3.13` if the interpreter
is not yet available. Tool configuration (ruff, mypy, pytest) lives in the
`[tool.*]` sections of `pyproject.toml`.

FaceFusion's `headless-run` pre-check requires the `ffmpeg` and `curl` system
binaries on `PATH` (install them with the OS package manager; uv does not
provide them). Without them every `/swap` job exits immediately.

## Quality Gates

Use these commands for project-owned code:

```bash
uv run python -m pytest -q
uv run ruff format --check config.py main_serve.py facefusion_job.py models.py tests conftest.py
uv run ruff check config.py main_serve.py facefusion_job.py models.py tests conftest.py
uv run mypy config.py main_serve.py facefusion_job.py models.py
```

The `[tool.mypy]` section of `pyproject.toml` intentionally excludes the
vendored FaceFusion tree. The `[tool.ruff]` section also excludes vendored files
and the launcher shims.

## Runtime Notes

- The API is mounted at `/v1/model/facefusion`.
- Startup uses `RAY_ADDRESS=auto` by default, so an existing Ray cluster is
  expected. Set `RAY_ADDRESS=` to let `main_serve.py` initialize Ray locally.
- `/swap` stores uploads, submits a `ray job submit --address=auto` subprocess,
  and returns immediately with `status="processing"`.
- Task status and logs are in-memory process state and do not survive restart.
- There is no authentication layer in this service.

## Documentation

Update `README.md`, `AGENTS.md`, `CLAUDE.md`, and local `.claude/skills/*`
runbooks when setup, runtime commands, test commands, or architecture change.
