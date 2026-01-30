# Model Configuration Loading Mechanism

## Modification History

| Date       | Summary of Changes |
|------------|-------------------|
| 2026-01-09 | Initial documentation of model config loading flow |

## Overview

`aiconfigurator` determines model architecture parameters (layers, hidden size, attention heads, etc.) through a prioritized three-stage lookup process. This ensures that:
1.  Common models work out-of-the-box (fast).
2.  Verified internal models are consistent (local cache).
3.  New models can be supported dynamically (online fallback).

## Lookup Flow

The resolution process happens in `src/aiconfigurator/sdk/models.py` via the `get_model()` function, which calls `_get_model_info()`.

```mermaid
graph TD
    A[Start: get_model("Name")] --> B{Check SupportedModels}
    B -- Found --> C[Return Hardcoded Params]
    B -- Not Found --> D{Check Local Cache}
    D -- Found --> E[Load model_configs/*.json]
    D -- Not Found --> F{Download from HF}
    F -- Success --> G[Parse HF config.json]
    F -- Failure --> H[Error]
```

### 1. Hardcoded Configuration (Highest Priority)
The system first checks `SupportedModels` in `src/aiconfigurator/sdk/common.py`. This dictionary maps model names directly to their architecture parameters.

**Example (`QWEN3_32B`):**
```python
"QWEN3_32B": [
    "LLAMA",  # Family
    64,       # Layers
    64,       # Heads
    8,        # KV Heads
    128,      # Head Dim
    5120,     # Hidden Size
    25600,    # Intermediate Size
    151936,   # Vocab Size
    40960,    # Context Length
    0, 0, 0, None # MoE/Extra Params
]
```

### 2. Local JSON Cache (Medium Priority)
If the model is not hardcoded, the system checks `CachedHFModels` in `common.py`. If a match is found, it loads the corresponding JSON file from `src/aiconfigurator/model_configs/`.

**Naming Convention:**
The HuggingFace ID `Organization/Model-Name` is mapped to the filename `Organization--Model-Name_config.json`.

**Example (`DeepSeek-V3`):**
*   **HF ID**: `deepseek-ai/DeepSeek-V3`
*   **File**: `src/aiconfigurator/model_configs/deepseek-ai--DeepSeek-V3_config.json`

### 3. HuggingFace Hub (Lowest Priority)
If no local definition exists, the tool attempts to fetch the `config.json` from the HuggingFace Hub using the provided string as the Model ID.

**Logic Location:**
*   `src/aiconfigurator/sdk/utils.py` -> `get_model_config_from_hf_id()`
*   Parses standard HF keys (e.g., `num_hidden_layers`, `hidden_size`) into the internal parameter list format.

## Key Code References

| Component | File Path | Responsibility |
|-----------|-----------|----------------|
| **Entry Point** | `sdk/models.py` | `get_model` initiates the lookup. |
| **Hardcoded Data** | `sdk/common.py` | Defines `SupportedModels` dictionary. |
| **Local Cache** | `model_configs/` | Directory containing cached JSON configs. |
| **HF Parser** | `sdk/utils.py` | `_parse_hf_config_json` converts generic JSON to internal format. |
