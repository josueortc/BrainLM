# Hugging Face Hub

## Loading a model

From the Hub (e.g. `josueortc/brainlm`):

```python
from transformers import AutoConfig, AutoModelForPreTraining

config = AutoConfig.from_pretrained("josueortc/brainlm", trust_remote_code=True)
model = AutoModelForPreTraining.from_pretrained("josueortc/brainlm", trust_remote_code=True)
```

`trust_remote_code=True` is required so the Hub can resolve the custom config and model classes via the repo’s `auto_map` in `config.json`.

Direct import (after `pip install` or cloning the repo):

```python
from brainlm_mae import BrainLMConfig, BrainLMForPretraining

config = BrainLMConfig.from_pretrained("josueortc/brainlm")
model = BrainLMForPretraining.from_pretrained("josueortc/brainlm", config=config)
```

## Pushing a checkpoint

From a local training run:

```python
from brainlm_mae import BrainLMForPretraining

model = BrainLMForPretraining.from_pretrained("./runs/my_run")
model.push_to_hub("josueortc/brainlm")
# Or with a specific revision:
# model.push_to_hub("josueortc/brainlm", revision="v0.1")
```

From the CLI (after logging in with `huggingface-cli login`):

```bash
cd runs/my_run
huggingface-cli upload josueortc/brainlm . .
```

Ensure the saved directory contains `config.json`, `pytorch_model.bin` (or `model.safetensors`), and optionally `training_args.bin`. The repo’s `config.json` must include `auto_map` so that `from_pretrained` with `trust_remote_code=True` loads the right classes.
