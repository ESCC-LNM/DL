# ESCC-LNM

This PyTorch model uses contrast-enhanced CT to predict metastasis in individual lymph nodes in esophageal squamous cell carcinoma. A shared ConvNeXt V2 Atto encoder processes the node and surrounding tissue. A learned radiomics prior guides feature modulation and sparse attention pooling.

## Quick start

Use Python 3.10 and install the PyTorch and torchvision builds for your hardware. From this directory, install the dependencies with `python -m pip install -r requirements.txt`.

Copy `configs/default.json` to `configs/local.json` and set the paths to the prepared CT volumes, node masks, radiomics tables, existing folds and scalers, and ImageNet initialization weights. The input formats are described in the [data guide](docs/DATA.md). The initialization weights and attribution are listed in the [third-party notices](THIRD_PARTY_NOTICES.md).

```bash
python validate_data.py --config configs/local.json
python train_cv.py --config configs/local.json
python train_fulltrain.py --config configs/local.json
```

Fold validation determines the full-training duration. Pooled out-of-fold predictions determine the classification threshold. Full-train uses that threshold to evaluate a single model on the held-out cohorts and saves `model.pt`. Run `python evaluate.py --help` to use that checkpoint for a separate cohort.

The manuscript describes five-fold probability averaging for held-out evaluation. Clinical data and weights are obtained separately.

Use only trusted checkpoints and scaler files. The pinned PyTorch release has a known checkpoint-loading vulnerability. [PyTorch advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6)
