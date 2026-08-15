# aila-models

Standalone model-provisioning package for AILA: the **catalog** of local
LLM / VLM / STT / sensory models and the **installer** that builds the
llama.cpp / whisper.cpp servers and downloads model weights.

This is deliberately separate from the `aila` runtime so different AILA
instances can share one catalog and vary models purely through configuration.

## Layout

```
aila-models/
├─ aila_models/            # installer package (code)
│  ├─ local_models.py      # `aila-local-models` entry point
│  └─ _paths.py
├─ catalog/
│  └─ local-models.toml    # the model catalog (single source of model facts)
└─ install-local-models.sh # operator-facing wrapper
```

## Usage

```bash
# From a checkout with the package installed (pip install -e .):
aila-local-models --config catalog/local-models.toml

# Or without installing the console script:
./install-local-models.sh
```

Models are declared in `catalog/local-models.toml` and referenced elsewhere by
**alias** — swapping a model is a one-line change here, with no Python edits.
