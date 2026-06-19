# CLAUDE.md

This file provides repository guidance for Claude Code when working with this
project. `AGENTS.md` contains the same project-level rules for other coding
agents.

## What This Is

FaceFusion Ray Serve API is a FastAPI service deployed through Ray Serve. It
wraps a vendored FaceFusion 3.6.1 snapshot and exposes `/swap`,
`/status/{task_id}`, `/health`, and `/stats` under `/v1/model/facefusion`.

Project-owned code is limited to the top-level service modules:

- `config.py`
- `main_serve.py`
- `facefusion_job.py`
- `models.py`
- `tests/`

The `facefusion/` directory, `facefusion.py`, and `install.py` are vendored
upstream FaceFusion code. Do not edit, format, lint, or type-check them unless
the task is explicitly to refresh the vendor snapshot.

## Environment

The project uses uv. Install all runtime and development dependencies with:

```bash
uv sync
```

Python 3.13 is managed by uv (run `uv python install 3.13` if needed). There is
no conda environment. Tool configuration (ruff, mypy, pytest) lives in the
`[tool.*]` sections of `pyproject.toml`.

The default execution provider is CUDA:

```bash
EXECUTION_PROVIDER=cuda
```

The GPU stack is pure PyPI: `onnxruntime-gpu` plus `nvidia-*` CUDA 13 wheels.
Because a uv virtualenv does not place those CUDA shared objects on the system
loader path, the service automatically injects `LD_LIBRARY_PATH` (pointing to
the venv's `nvidia/*/lib` directories) into the FaceFusion Ray job environment.
On a multi-node Ray cluster, every worker must have the same uv environment and
CUDA libraries present.

For CPU-only hosts, supply a CPU `onnxruntime` package and set:

```bash
EXECUTION_PROVIDER=cpu
```

Note that the default dependency set pins a CUDA build of `onnxruntime-gpu`; a
CPU host needs that replaced with a CPU `onnxruntime` package.

## Run

With an existing Ray cluster:

```bash
uv run ray start --head
uv run python main_serve.py
```

The default base URL is:

```text
http://0.0.0.0:9999/v1/model/facefusion
```

Runtime notes:

- `SERVICE_HOST` and `SERVICE_PORT` control Ray Serve HTTP binding.
- `RAY_ADDRESS=auto` expects an existing Ray cluster.
- Set `RAY_ADDRESS=` to let `main_serve.py` initialize Ray locally.
- `facefusion_job.py` submits FaceFusion through
  `ray job submit --address=auto -- python facefusion.py headless-run ...`.

## Quality Gates

```bash
uv run python -m pytest -q
uv run ruff format --check config.py main_serve.py facefusion_job.py models.py tests conftest.py
uv run ruff check config.py main_serve.py facefusion_job.py models.py tests conftest.py
uv run mypy config.py main_serve.py facefusion_job.py models.py
```

Use `ruff format` and `ruff check --fix` for project-owned files only. Do not
run bare `mypy .`; it would pull in the vendored FaceFusion tree.

## Architecture Notes

- FastAPI route functions live at module scope in `main_serve.py` so they are
  testable with `TestClient`; Ray Serve uses the same app through
  `@serve.ingress(app)`.
- `/swap` saves both uploaded files, derives an output path from the target file
  extension, schedules the Ray job, and returns immediately with
  `status="processing"`.
- Task status and logs live in in-memory dictionaries in `facefusion_job.py`.
  They do not survive process restart.
- There is no authentication or persistence layer in this service.
- A daemon cleanup thread removes uploads older than five days.
