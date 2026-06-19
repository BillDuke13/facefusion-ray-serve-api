---
name: serve
description: Start the FaceFusion Ray Serve service locally with the uv environment and smoke-test the API. Use when asked to run, start, launch, or verify the service is up.
---

Start the service and confirm it is serving, then hand control back. Do not block the session on the foreground process.

## Steps

1. **Install dependencies.** Ensure the uv virtualenv is current:
   ```bash
   uv sync
   ```

2. **Ray cluster.** Startup calls `ray.init(address="auto")` by default, which
   needs a running cluster. Either:
   - start one: `uv run ray start --head`, or
   - set `RAY_ADDRESS=` (empty) in `.env` so `main_serve.py` self-starts a local
     cluster.
   Check first with `uv run ray status`.

3. **Launch in the background** (the process runs forever on `while True`),
   capturing output:
   ```bash
   uv run python main_serve.py
   ```

4. **Wait for readiness.** Poll the health endpoint until it returns
   `{"status": "ok"}` (or time out after ~60s):
   ```bash
   curl -fsS http://localhost:9999/v1/model/facefusion/health
   ```
   The default port is **9999**. If `.env` overrides `SERVICE_PORT`, poll that
   port instead.

5. **Report.** Confirm the base URL `http://0.0.0.0:9999/v1/model/facefusion`
   and the endpoints (`POST /swap`, `GET /status/{task_id}`, `/health`,
   `/stats`). Tell the user how to stop it (kill the background process; the app
   shuts down gracefully on `KeyboardInterrupt`).

If startup fails, surface the relevant lines from the background process output
and the `logs/facefusion_service.log` file rather than guessing.
