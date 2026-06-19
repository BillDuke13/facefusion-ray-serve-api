---
name: typecheck
description: Run the strict mypy type-check gate for this repo and summarize failures. Use after editing Python in this project or before committing.
---

The `[tool.mypy]` section of `pyproject.toml` enforces strict typing
(`disallow_untyped_defs`, `disallow_untyped_calls`, `disallow_any_generics`).
Run it via uv and report concisely.

## Steps

1. Type-check the **project-owned modules only** — the vendored `facefusion/`
   package is upstream code and is not part of this gate:
   ```bash
   uv run mypy config.py main_serve.py facefusion_job.py models.py
   ```
   (Add any new top-level project module you introduce. Do not run bare
   `mypy .` — it pulls in the large vendored `facefusion/` tree.)

2. Summarize results: total errors, then group by file with the specific
   line/message. If clean, say so.

3. For any failure, propose the minimal typed fix (annotate signatures, narrow
   `Optional`, avoid bare `Any`) consistent with the existing code.
