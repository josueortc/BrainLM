# Installation

## Requirements

- Python ≥ 3.8
- PyTorch (CPU or CUDA)
- transformers ≥ 4.28
- datasets ≥ 2.0
- numpy, scikit-learn, matplotlib, pyarrow

Optional: `wandb` for experiment logging.

## Steps

```bash
git clone https://github.com/josueortc/BrainLM.git
cd BrainLM
pip install -r requirements.txt
```

Or install the package in development mode:

```bash
pip install -e .
```

## Verify

```bash
python -c "from brainlm_mae import BrainLMForPretraining; print('OK')"
pytest tests/ -v
```
