# CLAUDE.md

This file provides repository guidance for Claude Code when working with this
project. `AGENTS.md` contains the same project-level rules for other coding
agents.

## What This Is

FaceFusion Ray Serve API is a FastAPI service deployed through Ray Serve. It
wraps a vendored FaceFusion snapshot and exposes `/swap`, `/status/{task_id}`,
`/health`, and `/stats` under `/v1/model/facefusion`.

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

Run Python commands inside the conda environment:

```bash
conda activate facefusion-ray-serve-api
```

The environment targets Python 3.13 and a conda-forge stack. If `conda env
create -f environment.yml` is blocked by a channel Terms-of-Service prompt, use
the explicit `conda create --override-channels -c conda-forge ...` command from
`README.md`.

The default execution provider is CUDA:

```bash
EXECUTION_PROVIDER=cuda
```

For CPU-only hosts, install a CPU `onnxruntime` package and set:

```bash
EXECUTION_PROVIDER=cpu
```

## Run

With an existing Ray cluster:

```bash
ray start --head
python main_serve.py
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

Run the gates inside the conda environment:

```bash
conda run -n facefusion-ray-serve-api python -m pytest -q
conda run -n facefusion-ray-serve-api python -m ruff format --check config.py main_serve.py facefusion_job.py models.py tests conftest.py
conda run -n facefusion-ray-serve-api python -m ruff check config.py main_serve.py facefusion_job.py models.py tests conftest.py
conda run -n facefusion-ray-serve-api python -m mypy config.py main_serve.py facefusion_job.py models.py
```

Use `ruff format` and `ruff check --fix` for project-owned files only. Do not run
bare `mypy .`; it would pull in the vendored FaceFusion tree.

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
