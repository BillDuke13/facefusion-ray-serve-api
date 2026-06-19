# FaceFusion Ray Serve API

FaceFusion Ray Serve API is a FastAPI service deployed through Ray Serve. It
accepts source and target media uploads, submits a headless FaceFusion job to a
Ray cluster, and exposes task status, logs, health, and upload storage metrics.

The top-level Python modules are the service code. The `facefusion/` directory,
`facefusion.py`, and `install.py` are an upstream FaceFusion 3.6.1 snapshot and
should be treated as vendored code.

## Features

- FastAPI HTTP endpoints mounted under `/v1/model/facefusion`.
- Ray Serve deployment with asynchronous Ray job submission.
- Upload and output directories configured from `.env`.
- In-memory task status and per-task log tracking.
- Periodic cleanup for old uploads.
- Pytest, ruff, and strict mypy gates for project-owned modules.

## Requirements

- uv (manages the virtualenv and Python interpreter).
- Python 3.13 (installed by uv; run `uv python install 3.13` if needed).
- NVIDIA GPU with CUDA support for the default `EXECUTION_PROVIDER=cuda`.
  The project targets CUDA 13 (e.g. RTX 5090). GPU dependencies are
  `onnxruntime-gpu` plus `nvidia-*` CUDA 13 wheels from PyPI.
- CPU-only hosts can set `EXECUTION_PROVIDER=cpu`, but the default dependency
  set pins a CUDA build of `onnxruntime-gpu`; a CPU host needs a CPU
  `onnxruntime` package in its place.

## Quickstart

Clone the repository and enter it:

```bash
git clone https://github.com/BillDuke13/facefusion-ray-serve-api.git
cd facefusion-ray-serve-api
```

Install all runtime and development dependencies:

```bash
uv sync
```

The optional FaceFusion UI layer (gradio) is not installed by default; the
headless service never uses it. To include it:

```bash
uv sync --extra ui
```

Configure the service:

```bash
cp .env.example .env
```

Start a Ray cluster, then start the API:

```bash
uv run ray start --head
uv run python main_serve.py
```

For a self-started local Ray runtime, set `RAY_ADDRESS=` (empty) in `.env`
before running `uv run python main_serve.py`.

## Configuration

`.env` controls runtime behavior:

| Variable | Default | Description |
| --- | --- | --- |
| `UPLOAD_DIR` | `uploads` | Directory for incoming files. |
| `OUTPUT_DIR` | `outputs` | Directory for generated media. |
| `FACEFUSION_PATH` | `facefusion.py` | Headless FaceFusion launcher path. |
| `SERVICE_HOST` | `0.0.0.0` | Ray Serve HTTP host. |
| `SERVICE_PORT` | `9999` | Ray Serve HTTP port. |
| `RAY_ADDRESS` | `auto` | Existing Ray cluster address. Empty value starts Ray locally. |
| `EXECUTION_PROVIDER` | `cuda` | FaceFusion execution provider, usually `cuda` or `cpu`. |
| `LOG_LEVEL` | `DEBUG` | Intended log verbosity for local configuration. |

## API

The service is mounted at `http://localhost:9999/v1/model/facefusion` by
default.

Submit a face swap:

```bash
curl -X POST "http://localhost:9999/v1/model/facefusion/swap" \
  -H "accept: application/json" \
  -F "source_image=@source.jpg" \
  -F "target_image=@target.jpg"
```

Response:

```json
{
  "task_id": "7b55c9b8-5990-4f1e-8b41-25ad328ec731",
  "status": "processing",
  "output_path": "/path/to/outputs/7b55c9b8-5990-4f1e-8b41-25ad328ec731_output.jpg",
  "error": null
}
```

Check task status:

```bash
curl "http://localhost:9999/v1/model/facefusion/status/{task_id}"
```

Check health and storage stats:

```bash
curl "http://localhost:9999/v1/model/facefusion/health"
curl "http://localhost:9999/v1/model/facefusion/stats"
```

## Development

Run quality gates:

```bash
uv run python -m pytest -q
uv run ruff format --check config.py main_serve.py facefusion_job.py models.py tests conftest.py
uv run ruff check config.py main_serve.py facefusion_job.py models.py tests conftest.py
uv run mypy config.py main_serve.py facefusion_job.py models.py
```

Use `ruff format` and `ruff check --fix` on project-owned files before
committing changes. Do not format or lint the vendored `facefusion/` package.
Tool configuration (ruff, mypy, pytest) lives in `pyproject.toml`.

## Project Structure

```text
pyproject.toml        Project metadata, dependencies, and tool configuration.
uv.lock               Locked dependency graph (committed; do not edit by hand).
config.py             Runtime configuration and directory setup.
main_serve.py         FastAPI routes and Ray Serve deployment.
facefusion_job.py     Ray job submission and in-memory task tracking.
models.py             Pydantic response models.
tests/                Unit tests for project-owned modules.
facefusion/           Vendored upstream FaceFusion 3.6.1 code.
.claude/skills/       Local operational runbooks for this repository.
```

## GPU and CUDA note

The GPU stack is pure PyPI: `onnxruntime-gpu` plus `nvidia-*` CUDA 13 wheels.
Because a uv virtualenv does not place those CUDA shared objects on the system
loader path, the service automatically injects `LD_LIBRARY_PATH` (pointing to
the venv's `nvidia/*/lib` directories) into the FaceFusion Ray job environment.

On a multi-node Ray cluster, every worker node must have the same uv environment
and CUDA libraries present.

## Contributing

Keep changes scoped to project-owned modules unless intentionally refreshing the
vendored FaceFusion snapshot. Update tests and documentation for behavior,
configuration, or command changes.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](./LICENSE).
