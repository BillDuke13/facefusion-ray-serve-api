# FaceFusion Ray Serve API

FaceFusion Ray Serve API is a FastAPI service deployed through Ray Serve. It
accepts source and target media uploads, submits a headless FaceFusion job to a
Ray cluster, and exposes task status, logs, health, and upload storage metrics.

The top-level Python modules are the service code. The `facefusion/` directory,
`facefusion.py`, and `install.py` are an upstream FaceFusion snapshot and should
be treated as vendored code.

## Features

- FastAPI HTTP endpoints mounted under `/v1/model/facefusion`.
- Ray Serve deployment with asynchronous Ray job submission.
- Upload and output directories configured from `.env`.
- In-memory task status and per-task log tracking.
- Periodic cleanup for old uploads.
- Pytest, ruff, and strict mypy gates for project-owned modules.

## Requirements

- Conda.
- Python 3.13 in the project environment.
- NVIDIA GPU with CUDA support for the default `EXECUTION_PROVIDER=cuda`.
- CPU-only hosts can set `EXECUTION_PROVIDER=cpu` and use a CPU
  `onnxruntime` package.

## Quickstart

Clone the repository and enter it:

```bash
git clone https://github.com/BillDuke13/facefusion-ray-serve-api.git
cd facefusion-ray-serve-api
```

Create the environment from conda-forge. If `conda env create -f
environment.yml` is blocked by a channel Terms-of-Service prompt, create the
same stack explicitly with `--override-channels`:

```bash
conda create -n facefusion-ray-serve-api --override-channels -c conda-forge \
  python=3.13 pip setuptools wheel numpy scipy onnx 'onnxruntime=*=*cuda*' \
  cuda-libraries cudnn opencv tqdm 'ray-serve>=2.55' fastapi uvicorn python-dotenv
conda run -n facefusion-ray-serve-api pip install \
  filetype 'gradio>=5,<6' gradio-rangeslider python-multipart python-jose ruff pytest
conda activate facefusion-ray-serve-api
```

Configure the service:

```bash
cp .env.example .env
```

Start a Ray cluster, then start the API:

```bash
ray start --head
python main_serve.py
```

For a self-started local Ray runtime, set `RAY_ADDRESS=` in `.env` before
running `python main_serve.py`.

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

Run quality gates inside the conda environment:

```bash
conda run -n facefusion-ray-serve-api python -m pytest -q
conda run -n facefusion-ray-serve-api python -m ruff format --check config.py main_serve.py facefusion_job.py models.py tests conftest.py
conda run -n facefusion-ray-serve-api python -m ruff check config.py main_serve.py facefusion_job.py models.py tests conftest.py
conda run -n facefusion-ray-serve-api python -m mypy config.py main_serve.py facefusion_job.py models.py
```

Use `ruff format` and `ruff check --fix` on project-owned files before
committing changes. Do not format or lint the vendored `facefusion/` package.

## Project Structure

```text
config.py             Runtime configuration and directory setup.
main_serve.py         FastAPI routes and Ray Serve deployment.
facefusion_job.py     Ray job submission and in-memory task tracking.
models.py             Pydantic response models.
tests/                Unit tests for project-owned modules.
facefusion/           Vendored upstream FaceFusion code.
.claude/skills/       Local operational runbooks for this repository.
```

## Contributing

Keep changes scoped to project-owned modules unless intentionally refreshing the
vendored FaceFusion snapshot. Update tests and documentation for behavior,
configuration, or command changes.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](./LICENSE).
