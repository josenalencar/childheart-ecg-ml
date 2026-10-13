# Analysis pipeline

Run in numerical order. `01` writes the tuned hyperparameters that every later script reads, so
it must run first; `11` depends on `01`, `08` and `10`.

Each script expects, beside itself, a `data/` directory with the modelling dataset and the
external feature table, and writes to `out/<topic>/`. Nothing is written back into `data/`.

| script | what it produces |
|---|---|
| `prep_dataset.py` | the modelling dataset: cleaned features, age-z normalisation, fixed 80/20 split |
| `01_ceiling_models.py` | nested cross-validation of three estimator families; tuned hyperparameters; held-out test predictions |
| `02_calibration_lottery_riley_subgroups.py` | calibration, sample size against the Riley criteria, subgroups, split lottery (random forest) |
| `03_shap_stability_parsimony.py` | importance, bootstrap stability, parsimony floor (random forest) |
| `04_delong.py` | pairwise DeLong comparisons of the estimators |
| `05_ceiling_external.py` | the frozen models applied to the external cohort, by case definition |
| `08_external_calibration_dca.py` | calibration internally and externally; decision curves |
| `10_manifesting_three_estimators.py` | ECG-manifesting lesion subanalysis for all three estimators |
| `11_lightgbm_primary.py` | **the primary-model analyses**: operating characteristics, subgroups, split lottery, calibration, SHAP, stability, parsimony floor, decision curves, transportable-feature retrain |

`02` and `03` are retained as published because they document the random-forest comparator;
their LightGBM equivalents, which the paper reports, are in `11`.

## Shared modules

| module | role |
|---|---|
| `common_features.py` | feature-matrix assembly, implausible-value cleaning, age-z transform |
| `batch_source.py` | the single source of truth for which extraction vintage is read (`CH_VINTAGE`) |
| `engine_provenance.py` | records the measurement-engine commit that produced the feature tables |

`external_validation/` builds the external cohort and holds the lesion-list definitions.
