# Data setup

Copy `configs/default.json` to `configs/local.json` and set the paths for `train`, `internal`, `external1` and `external2`.

## Cohort files

- **CSV** with one row per lymph node, a unique `ID`, a binary `label` and the 45 [radiomics columns](../protocol/feature_names_cv.json). Each cohort needs both labels (0 and 1). Use the same `patientID` for nodes from the same patient.
- **CT and mask** for each `ID`. Supply preprocessed, aligned 3D CT regions and binary masks. For example, the image and mask for `case_A` can both be named `case_A.nii.gz` in their respective directories.

## Cross-validation

Set `paths.protocol` to the existing five-fold JSON. Each fold (`Fold_1` through `Fold_5`) needs `train_ids`, `val_ids`, `selected_features` in the same order across folds, and `scaler_file`. Place each fitted training scaler next to the JSON. Each fold partitions the training pool by patient, and each node appears in validation exactly once across all folds.

```bash
python validate_data.py --config configs/local.json
```
