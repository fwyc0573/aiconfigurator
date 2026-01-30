# Task Plan: Enhance Test Coverage and Documentation

## Goal
Expand test coverage for parallelism strategies, audit existing tests, explain model config loading, and provide a Web UI access guide.

## Phases
- [ ] Phase 1: Investigation
    - [ ] Trace model config loading mechanism (`sdk/models.py` -> ?)
    - [ ] Analyze Web UI launch arguments and structure
    - [ ] Audit existing scripts for missing features
- [x] Phase 2: Test Enhancement (Tasks 1 & 2)
    - [x] Create `test_parallelism_coverage.sh`
    - [x] Create `coverage_exp.yaml` (MoE, TP/PP/EP variations)
    - [x] Document test gap analysis
- [x] Phase 3: Model Config Documentation (Task 3)
    - [x] Write `model_config_loading.md`
- [x] Phase 4: Web UI Guide (Task 4)
    - [x] Test Web UI launch
    - [x] Write `web_ui_guide.md` (SSH forwarding)

## Key Questions
1. Where are model params (layers, hidden size) stored? (Resolved: Hardcoded + `model_configs/`)
2. Does Web UI support custom ports/host binding? (Resolved: Yes, via `--server_name` / `--server_port`)

## Status
**Completed.** All tasks finished. Deliverables in `tests/baselines/aiconfigurator/`.
