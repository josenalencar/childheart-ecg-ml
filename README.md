# CHILDHEART ECG machine learning

Analysis code for **"Machine Learning Detects Pediatric Structural Heart Disease From the
12-Lead Electrocardiogram: External Validation in 10,352 Children"** (submitted to *Pediatric
Cardiology*).

Models are derived in 1,637 Brazilian children whose chamber involvement was adjudicated by
three pediatric cardiologists, and applied frozen to 10,352 Chinese children labelled by an
ICD-10 crosswalk. The study varies two things independently — the learning algorithm and the
external case definition — and finds that the internal-to-external gap is essentially
unchanged across three estimator families (0.285 / 0.290 / 0.284) while restricting cases to
lesions that load or injure the heart recovers external discrimination from 0.68 to 0.91.

## Layout

```
analysis/                 the pipeline, run in numerical order
  prep_dataset.py         modelling dataset: cleaning, age-z normalisation, fixed 80/20 split
  01…05, 08, 10, 11       estimator comparison, DeLong, external application, calibration,
                          decision curves, lesion subanalysis, primary-model analyses
  common_features.py      feature assembly, implausible-value cleaning, age-z transform
  batch_source.py         single source of truth for the extraction vintage
  engine_provenance.py    records the measurement-engine commit behind the feature tables
  external_validation/    external cohort build and the lesion-list definitions
manuscript/               scripted manuscript builder; nums.py reads every number in the paper
                          from the result files, so no value is ever typed into prose
docs/                     how the submission package is assembled
```

`analysis/README.md` lists what each script produces and the order they depend on.

## Reproducing

The analyses expect, next to the scripts, a `data/` folder with the modelling dataset and the
external feature table, and write to `out/`. Run `01_ceiling_models.py` first (it writes the
tuned hyperparameters every later script reads), then the rest in numerical order.

```bash
python3 -m pip install numpy pandas scikit-learn xgboost lightgbm shap statsmodels openpyxl joblib
python3 code/tripod_analyses/01_ceiling_models.py
python3 code/tripod_analyses/11_lightgbm_primary.py
```

Feature extraction depends on our `ecg-vcg-viewer` engine at delineator D2, extractor E3 and
measurement contract M2, commit `150be8b`. That engine is not public; it is available from the
authors on reasonable request. Because measured values move between extractor versions, the
commit is part of the method — running these scripts against another commit will not reproduce
the published numbers exactly.

## Data

**No patient data are in this repository.** The derivation data contain potentially identifying
pediatric clinical information and are available from the corresponding author on reasonable
request, subject to research-ethics approval and a data-use agreement. The external cohort is
the published Zhengzhou University pediatric ECG database (Sci Data 2025;
[10.1038/s41597-025-05225-z](https://doi.org/10.1038/s41597-025-05225-z)), deposited by its
authors on figshare and used here under its stated terms.

## Ethics

The derivation cohort was approved by the institutional review board of Instituto Dante
Pazzanese de Cardiologia (protocol 75263023.5.0000.5462), which waived informed consent for
this retrospective analysis of de-identified records.

## Licence

Code released under the MIT licence (see `LICENSE`).
