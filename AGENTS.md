# AI Agent Configuration (`AGENTS.md`)

This file provides context, commands, and guidelines for AI agents (like Sisyphus, Cursor, Copilot) operating in the `aiconfigurator` repository.

## 1. Project Overview

`aiconfigurator` is a tool for offline configuration of disaggregated LLM serving. It models inference performance to find optimal deployment configurations (TP/PP/EP, worker counts) for Dynamo.

- **Core Language**: Python (>=3.9)
- **Web UI**: Python (Gradio) + JavaScript
- **Key Frameworks**: FastAPI, Uvicorn, Pandas, Pytest
- **Linting/Formatting**: Ruff (Python), ESLint (JS)

## 2. Environment & Build

### Installation
```bash
# Development setup (install dev, webapp, and service dependencies)
pip install -e ".[dev,webapp,service]"

# Install pre-commit hooks (CRITICAL for all contributions)
pre-commit install
```

### Build (Docker)
```bash
docker build -f docker/Dockerfile --no-cache --target build -t aiconfigurator:latest .
```

## 3. Verification Commands

**Always run these verification steps before finishing a task.**

### Linting & Formatting (Ruff)
The project uses strict linting rules defined in `pyproject.toml`.
```bash
# Check for issues
ruff check .

# Auto-fix fixable issues (imports, formatting, etc.)
ruff check --fix .

# Format code
ruff format .
```

### Pre-commit (Run Manually)
```bash
pre-commit run --all-files
```

### Testing (Pytest)
```bash
# Run all tests
pytest tests

# Run unit tests only (fastest feedback loop)
pytest tests/unit

# Run a specific test file
pytest tests/unit/test_main.py

# Run a specific test case
pytest tests/unit/test_main.py::test_cli_default

# Run "build" subset (unit + small stable E2E) - use for quick regression check
pytest -m "unit or build"
```

## 4. Code Style & Conventions

### Python
- **Style**: PEP 8 enforced by Ruff.
- **Imports**: Sorted automatically by Ruff (`isort` rules).
  - **First-party modules**: `aiconfigurator`, `utils`, `helper`
  - Grouping: Stdlib -> Third-party -> First-party
- **Type Hints**: **Mandatory**. Use Python 3.9+ syntax.
  - Add `from __future__ import annotations` to all files.
  - Use `list[str]`, `dict[str, Any]` instead of `List`, `Dict`.
- **Naming**:
  - Classes: `PascalCase` (`ModelConfig`, `InferenceSession`)
  - Functions/Variables: `snake_case` (`get_model_info`, `total_gpus`)
  - Constants: `UPPER_SNAKE_CASE` (`DEFAULT_TIMEOUT`)
- **Docstrings**: Google style.
- **Error Handling**: Fail fast. Use explicit exceptions. Avoid bare `except:`.

### JavaScript
- **Style**: Enforced by `eslint.config.js`.
- **Key Rules**:
  - No semicolons (`semi: ["error", "never"]`)
  - Double quotes (`quotes: ["error", "double"]`)
  - `const` over `let` where possible (`prefer-const`)
  - `===` over `==` (`eqeqeq`)

### License Headers
All new files **MUST** include the SPDX license header:
```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

## 5. Agent Behavior Rules (from System Config)

### Interaction Protocol
- **Salutation**: Every response must start with **"Yes, boss Yicheng!"**
- **Language**:
  - **Explanations**: Clear, professional **Chinese** (中文).
  - **Code/Comments**: Standard **English**.
  - **Technical Terms**: Keep in **English** (e.g., token, prefill, decode, TPOT, TTFT).

### Task Management
- **Complex Tasks**: If a task involves multiple files or steps, use the `todowrite` tool to track progress.
- **File Creation**:
  - Experimental scripts -> `tests/` (subdirectories: `unit/`, `integration/`, etc.).
  - **NEVER** clutter the root directory with temp files.
- **Documentation**: Update existing `.md` files (add to "Modification History") instead of creating duplicates.

### Error Handling
- **No Fallbacks**: Do not implement silent fallbacks that hide bugs.
- **Fail Fast**: Raise explicit errors for unexpected conditions.
