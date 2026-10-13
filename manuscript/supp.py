"""Tables, figure legends and the supplement, all generated from the result files.

Three main tables and two figures, which is as compact as this study can honestly be: the
cohorts (because the case-control assembly is the main threat to validity), the primary
model's internal performance, and the estimator-by-case-definition grid that carries the
argument. Everything else -- tuning, DeLong, calibration, subgroups, the split lottery, SHAP,
stability, parsimony, transportability, Riley, chamber status, decision curves -- goes to the
supplement as Online Resource 1, in the order the Results mention it.

Rows are built by looping over the CSVs, so a table cannot fall out of step with the text.
"""
import nums as N

N_TABLES, N_FIGURES = 3, 2

# ------------------------------------------------------------------ Table 1: the cohorts
T1 = [
  ['Children analysed, n', N.comma(N.N_INTERNAL), N.comma(N.N_EXT)],
  ['Cases, n (%)', f"{N.N_CASES} ({N.pct(N.CASE_FRACTION)})",
                   f"{N.comma(N.N_EXT_EVENTS)} ({N.pct(N.N_EXT_EVENTS / N.N_EXT)})"],
  ['Comparison group, n', N.comma(N.N_CONTROLS), N.comma(int(N.mani_row('ANY disease')['n_disease_free']))],
  ['Median age, months (IQR)', '51 (15-105)', '114 (69-153)'],
  ['Aged <3 years, %', '39.0', '9.2'],
  ['Male, %', '50.0', '57.7'],
  ['Reference standard', 'Adjudication by three pediatric cardiologists '
                         '(echocardiography, clinical, imaging and surgical records)',
                         'ICD-10 diagnostic code crosswalk'],
  ['Cases with prior cardiac surgery, %', str(N.PCT_POSTSURGICAL), 'Not recorded'],
  ['Role in this study', f'Derivation; fixed 80/20 split '
                         f'({N.comma(N.N_TRAIN)} train, {N.N_TEST} held-out test)',
                         'External validation; frozen models, three case definitions'],
]
TABLE1 = dict(
  title='Table 1 Characteristics of the derivation and external validation cohorts',
  head=['Characteristic', 'Derivation (CHILDHEART, Brazil)', 'External validation (ZZU, China)'],
  rows=T1, widths=[1.6, 2.4, 2.4],
  note=f'IQR interquartile range, ICD-10 International Classification of Diseases Tenth '
       f'Revision, ZZU Zhengzhou University. Cases and the comparison group were recruited '
       f'separately in the derivation cohort, so the case fraction of '
       f'{N.pct(N.CASE_FRACTION)}% is fixed by design and is not an estimate of prevalence; '
       f'predictive values therefore do not transfer to clinical populations. The external '
       f'comparison group shown is the disease-free group used for the ECG-manifesting '
       f'analysis.')

# -------------------------------------- Table 2: the primary model, internal held-out
T2 = []
for o in N.OUTCOMES:
    t = N.test_auc(o); y = N.ops(o, 'Youden (OOF train)'); s = N.ops(o, '>=95% specificity (OOF train)')
    T2.append([
        o.replace('ANY disease', 'Any structural disease'),
        f"{int(y['events_test'])}/{int(y['n_test'])}",
        f"{N.f3(t['test_auc'])} ({N.f3(t['ci_lo'])}-{N.f3(t['ci_hi'])})",
        f"{N.f3(y['pr_auc'])}",
        f"{N.f3(y['brier'])}",
        f"{N.f3(y['threshold'])}",
        f"{N.pct(y['sensitivity'])} / {N.pct(y['specificity'])}",
        f"{N.f2(y['lr_pos'])}",
        f"{N.pct(s['specificity'])} / {N.pct(s['sensitivity'])}",
        f"{N.f2(s['lr_pos'])}",
    ])
TABLE2 = dict(
  title='Table 2 Discrimination and operating characteristics of the primary model (LightGBM) '
        'in the internal held-out test set',
  head=['Outcome', 'Events/n', 'ROC-AUC (95% CI)', 'PR-AUC', 'Brier', 'Youden threshold',
        'Sens / Spec, %', 'LR+', 'Spec / Sens at >=95% spec, %', 'LR+'],
  rows=T2, widths=[1.1, 0.6, 1.1, 0.5, 0.45, 0.6, 0.75, 0.4, 0.8, 0.4],
  note='ROC-AUC area under the receiver operating characteristic curve, PR-AUC area under the '
       'precision-recall curve, Sens sensitivity, Spec specificity, LR+ positive likelihood '
       'ratio. Both thresholds were fixed on out-of-fold predictions within the training '
       'partition and applied unchanged to the held-out test set; neither was chosen on the '
       'test data. Confidence intervals are 2,000-sample bootstrap percentiles. Because the '
       'case fraction was fixed by design, PR-AUC and predictive values describe the '
       'derivation sample only, and likelihood ratios are the transportable quantities.')

# ------------------- Table 3: the estimator-by-case-definition grid (the argument)
# the forest as shipped was never applied externally, so it has no external row
EXT_EST = [N.RF_TUNED, N.XGB, N.PRIMARY]
EST_ORDER = [(N.RF_SHIPPED, 'Random forest (parent analyses)'),
             (N.RF_TUNED, 'Random forest (tuned)'),
             (N.XGB, 'XGBoost'),
             (N.PRIMARY, 'LightGBM (primary)')]
T3 = []
for est, label in EST_ORDER:
    t = N.test_auc('ANY disease', est)
    try:
        e = N.ext_row('ANY disease', est); m = N.mani_row('ANY disease', est)
        ext, ns, gap = e['external_all_icd_auc'], e['external_nonshunt_auc'], e['internal_minus_external_all']
        mani = m['manifesting_auc']
    except Exception:
        ext = ns = gap = mani = None
    T3.append([label,
               f"{N.f3(N.nested('ANY disease', est))}",
               f"{N.f3(t['test_auc'])} ({N.f3(t['ci_lo'])}-{N.f3(t['ci_hi'])})",
               N.f3(ext) if ext is not None else '-',
               N.f3(ns) if ns is not None else '-',
               N.f3(mani) if mani is not None else '-',
               N.f3(gap) if gap is not None else '-'])
TABLE3 = dict(
  title='Table 3 Internal and external discrimination for any structural disease, by estimator '
        'and by external case definition',
  head=['Estimator', 'Nested CV (train only)', 'Internal held-out (95% CI)',
        'External, all ICD-10 codes', 'External, non-shunt', 'External, ECG-manifesting',
        'Internal minus external (all codes)'],
  rows=T3, widths=[1.3, 0.8, 1.2, 0.85, 0.7, 0.8, 0.85],
  note=f'CV cross-validation, ICD-10 International Classification of Diseases Tenth Revision. '
       f'The primary estimator was chosen on nested cross-validation within the training '
       f'partition alone (column 2). All models were then applied frozen to the external '
       f'cohort; only the case definition changes across columns 4 to 6. The final column is '
       f'the quantity of interest: it is essentially constant across estimators '
       f'({N.f3(min(float(N.ext_row("ANY disease", e)["internal_minus_external_all"]) for e in EXT_EST))}'
       f'-{N.f3(max(float(N.ext_row("ANY disease", e)["internal_minus_external_all"]) for e in EXT_EST))}), '
       f'so a stronger algorithm raises both ends and closes none of the gap, whereas changing '
       f'the case definition recovers most of it. The non-shunt definition excludes children '
       f'whose only coded lesion was an isolated atrial or ventricular septal defect or a '
       f'patent ductus arteriosus; the ECG-manifesting definition uses pre-specified lesion '
       f'lists (Online Resource 1, eMethods 3).')

TABLES = [TABLE1, TABLE2, TABLE3]

# ------------------------------------------------------------------ figure legends
_m = N.mani_row('ANY disease')
FIGURE_LEGENDS = [
  f"**Fig. 1** Flow of participants through both cohorts. **a** The derivation cohort was "
  f"assembled as a diagnostic case-control study: cases were consecutive admissions whose "
  f"chamber involvement was adjudicated by three pediatric cardiologists blinded to the ECG, "
  f"and the comparison group was recruited separately from tele-electrocardiography services "
  f"serving general care and did not undergo echocardiography, so the case fraction of "
  f"{N.pct(N.CASE_FRACTION)}% is fixed by design. **b** The external cohort was labelled by an "
  f"ICD-10 crosswalk; the models were applied frozen, and the three pre-specified case "
  f"definitions differ only in what counts as a case. Shading marks how the reference standard "
  f"was obtained, and the ROC-AUC shown for each external definition is that of the primary "
  f"model (LightGBM) against the {int(_m['n_disease_free']):,} disease-free children. "
  f"ECG electrocardiogram, ICD-10 International Classification of Diseases Tenth Revision, "
  f"ROC-AUC area under the receiver operating characteristic curve",

  f"**Fig. 2** Discrimination for any structural heart disease by estimator and by external "
  f"case definition. Points are ROC-AUC with 95% confidence intervals; the four estimators are "
  f"shown within each panel. Moving from left to right, the models are unchanged and only the "
  f"reference standard changes. The estimators separate internally and move together "
  f"externally, and the internal-to-external difference is essentially identical across them "
  f"(right panel), whereas restricting cases to lesions expected to alter the ECG recovers "
  f"discrimination to {N.f3(_m['manifesting_auc'])} for every estimator. ROC-AUC area under "
  f"the receiver operating characteristic curve, ICD-10 International Classification of "
  f"Diseases Tenth Revision",
]

# ------------------------------------------------------------------ supplement: eMethods
EMETHODS = [
 ("eMethods 1. Electrocardiographic feature extraction and age normalization", [
  f"Tracings were processed by the ecg-vcg-viewer pipeline at delineator D2, extractor E3 and "
  f"measurement contract M2, commit {N.ENGINE_COMMIT} (feature tables v4_E3). Measured values "
  f"move between extractor versions, so the commit is part of the method: re-running these "
  f"analyses on another commit will not reproduce the numbers exactly. The extractor emitted "
  f"1,107 raw parameters per tracing.",
  f"Biologically impossible values were set to missing and imputed with training-set medians. "
  f"The implausibility thresholds were heart rate outside 25-320 beats/min, PR outside "
  f"50-400 ms, QRS outside 30-250 ms, QT outside 150-800 ms, QTc outside 200-700 ms, P-wave "
  f"duration outside 10-220 ms, and absolute amplitude above 8 mV. After filtering, "
  f"{N.comma(N.N_FEATURES)} features remained.",
  f"Continuous features were converted to age-specific z scores over eight partitions spanning "
  f"0-300 months. Normalization constants were estimated only among disease-free children "
  f"inside the training fold, interpolated linearly between partition midpoints and truncated "
  f"at +/-8 SD. Binary, ordinal and count features and age in months were left raw. The fitted "
  f"matrix was applied to the external cohort without re-estimation, so no external information "
  f"entered the models.",
 ]),
 ("eMethods 2. Estimator tuning and selection of the primary model", [
  f"Three estimator families were fitted to the same training partition, feature matrix and "
  f"seed (42): a random forest, XGBoost and LightGBM. Each was tuned by randomized search over "
  f"15 draws with three-fold inner cross-validation, nested inside a five-fold outer loop, so "
  f"the cross-validated estimate carries no tuning optimism. The random forest was additionally "
  f"fitted with the parent CHILDHEART hyperparameters (600 trees, balanced class weights, "
  f"minimum leaf size 2) to separate an estimator effect from an untuned baseline.",
  f"The primary estimator was selected on training data alone, by the highest nested "
  f"cross-validated ROC-AUC for the primary outcome: LightGBM "
  f"{N.f3(N.nested('ANY disease'))}, XGBoost {N.f3(N.nested('ANY disease', N.XGB))}, tuned "
  f"random forest {N.f3(N.nested('ANY disease', N.RF_TUNED))}. The same family was then used "
  f"for all five outcomes rather than selecting per outcome; eTable 2 reports the nested "
  f"cross-validated values for every outcome and estimator, including the one outcome (left "
  f"atrial) for which LightGBM was the weakest of the three.",
  f"Thresholds were fixed on out-of-fold predictions inside the training partition, at the "
  f"Youden point and at the lowest threshold reaching 95% specificity, and transferred "
  f"unchanged to the held-out test set. Tuned hyperparameters are in eTable 1.",
 ]),
 ("eMethods 3. External case definitions and lesion lists", [
  f"Three pre-specified definitions were applied to the same frozen models, each against "
  f"disease-free children. (1) All ICD-10: every child carrying a cardiac ICD-10 code "
  f"({N.comma(N.N_EXT_EVENTS)} of {N.comma(N.N_EXT)}). (2) Non-shunt: children with a "
  f"CHILDHEART-type congenital lesion that was not an isolated atrial septal defect, "
  f"ventricular septal defect or patent ductus arteriosus, and, for a chamber outcome, also "
  f"carrying that chamber's ICD-10 flag. (3) ECG-manifesting: cases restricted to lesions "
  f"expected on physiological grounds to load or injure the chamber directly, using positive "
  f"lesion lists fixed before analysis.",
  "The ECG-manifesting lists were: any disease, complex congenital heart disease, pulmonary "
  "hypertension and tetralogy of Fallot; left ventricle, dilated and hypertrophic and other "
  "cardiomyopathies, aortic stenosis and coarctation; right ventricle, tetralogy of Fallot, "
  "pulmonary stenosis and pulmonary hypertension; right atrium, tricuspid valve disease, "
  "pulmonary hypertension and complex congenital disease; left atrium, mitral valve disease, "
  "atrioventricular septal defect and complex congenital disease. Shunts are deliberately "
  "absent from every list.",
  "The left atrial list was revised once during preliminary analysis, from mitral disease, "
  "ventricular septal defect and patent ductus arteriosus to the direct loading lesions above, "
  "because the initial set captured 95.7% of cases non-specifically. This is disclosed as a "
  "post-hoc change.",
 ]),
 ("eMethods 4. Calibration, clinical usefulness, transportability and stability", [
  "Calibration was summarized by calibration-in-the-large (the intercept of a logistic model "
  "fitted with the linear predictor as a fixed offset) and the calibration slope (the "
  "coefficient on the linear predictor), with the Brier score. Clinical usefulness was assessed "
  "by decision-curve analysis over threshold probabilities of 0.01 to 0.50, comparing the "
  "model's net benefit with imaging every child and imaging no child.",
  "Feature transportability was assessed by computing a direction-agnostic univariate ROC-AUC "
  "for every feature in the internal training partition and in the external cohort. To avoid "
  "selecting features and reporting performance in the same children, the external cohort was "
  "split once at patient level, stratified by disease status (seed 42), into a selection half "
  "and a reporting half; a feature counted as transportable when its univariate ROC-AUC reached "
  "0.55 in both the internal training partition and the external selection half, and every "
  "external estimate for the refitted models was computed in the reporting half only.",
  f"Ranking stability was estimated by refitting on "
  f"{int(N.stab('ANY disease')['n_resamples'])} bootstrap resamples of the training partition "
  f"and recording how often each feature re-entered the top twenty. The parsimony floor "
  f"compared penalized logistic regression and LightGBM restricted to the five, ten and twenty "
  f"most stable features against the full model. Because a single held-out estimate is one "
  f"draw, the outcome-stratified split was repeated "
  f"{int(N.lot('ANY disease')['n_splits'])} times; that comparison reuses hyperparameters tuned "
  f"on the original training partition, so it describes split-to-split variability rather than "
  f"a fully nested re-derivation.",
 ]),
]

# ------------------------------------------------------------------ supplement: eTables
def _et(title, head, rows, note=''):
    return dict(title=title, head=head, rows=rows, note=note)

_hp = N.LG_HP.copy()
ETABLES = [
 _et('eTable 1. Tuned LightGBM hyperparameters, by outcome',
     list(_hp.columns),
     [[('' if str(v) == 'nan' else (f'{v:.4g}' if isinstance(v, float) else str(v)))
       for v in r] for r in _hp.itertuples(index=False)],
     'Selected by randomized search with three-fold inner cross-validation inside a five-fold '
     'outer loop, using training data only.'),

 _et('eTable 2. Nested cross-validated ROC-AUC by outcome and estimator (training partition only)',
     ['Outcome'] + [lab for _, lab in EST_ORDER],
     [[o] + [N.f3(N.nested(o, est)) for est, _ in EST_ORDER] for o in N.OUTCOMES],
     'These are the values the primary-model selection rule acted on. LightGBM was highest for '
     'the primary outcome of any structural disease and was therefore used for all five '
     'outcomes; it was the weakest of the three for left atrial disease, which is disclosed in '
     'the limitations.'),

 _et('eTable 3. Pairwise comparison of estimators by DeLong\'s test, internal held-out test set',
     ['Outcome', 'Model A', 'AUC A', 'Model B', 'AUC B', 'Difference (95% CI)', 'P'],
     [[r.outcome, r.model_A, N.f3(r.auc_A), r.model_B, N.f3(r.auc_B),
       f'{N.f3(r.auc_difference)} ({N.f3(r.diff_ci_lo)} to {N.f3(r.diff_ci_hi)})',
       ('<0.001' if float(r.delong_p) < 0.001 else f'{float(r.delong_p):.3f}')]
      for r in N.DELONG.itertuples(index=False)],
     'DeLong\'s test for two correlated receiver operating characteristic curves.'),

 _et('eTable 4. Calibration of the primary model internally and externally',
     ['Outcome', 'Cohort', 'n', 'Events', 'Observed prevalence', 'Mean predicted',
      'Calibration-in-the-large', 'Calibration slope', 'Brier', 'ROC-AUC'],
     [[r.outcome, r.cohort, N.comma(r.n), N.comma(r.events), N.f3(r.observed_prevalence),
       N.f3(r.mean_predicted), N.f2(r.calibration_in_the_large), N.f2(r.calibration_slope),
       N.f3(r.brier), N.f3(r.auc)]
      for r in N.CAL[N.CAL.estimator == N.PRIMARY].itertuples(index=False)],
     'A calibration slope above 1 indicates predictions compressed toward the case fraction; '
     'below 1, predictions too extreme. A negative calibration-in-the-large indicates '
     'systematic over-prediction of risk. Discrimination transported across cohorts; '
     'calibration did not.'),

 _et('eTable 5. Subgroup discrimination of the primary model in the internal held-out test set',
     ['Outcome', 'Subgroup', 'n', 'Events', 'ROC-AUC (95% CI)'],
     [[r.outcome, r.subgroup, int(r.n), int(r.events),
       ('not estimable' if str(r.auc) == 'nan' else f'{N.f3(r.auc)} ({N.f3(r.auc_lo)}-{N.f3(r.auc_hi)})')]
      for r in N.LG_SUB.itertuples(index=False)],
     'Event counts within strata are small and these comparisons are descriptive. Ethnicity was '
     'not recorded, so fairness across ethnic groups could not be assessed.'),

 _et('eTable 6. Distribution of held-out discrimination over repeated random splits',
     ['Outcome', 'Splits', 'Mean', 'SD', '2.5th pct', '97.5th pct', 'Min', 'Max',
      'Reported fixed split', 'Percentile of reported split'],
     [[r.outcome, int(r.n_splits), N.f3(r.mean), N.f3(r.sd), N.f3(r.p2_5), N.f3(r.p97_5),
       N.f3(r.min), N.f3(r.max), N.f3(r.reported_fixed_split), f'{r.percentile_of_reported:.0f}']
      for r in N.LG_LOT.itertuples(index=False)],
     'The fixed split reported throughout the manuscript sits in the lower part of its own '
     'distribution for most outcomes, so the headline figures are a conservative draw. This '
     'comparison reuses hyperparameters tuned on the original training partition and is to that '
     'extent optimistic about the distribution itself.'),

 _et('eTable 7. Parsimony floor: small models against the full feature set',
     ['Outcome', 'k', 'Logistic regression, CV', 'Logistic regression, test',
      'LightGBM on k features, test', 'LightGBM on all features, test',
      '% of chance-corrected AUC recovered'],
     [[r.outcome, int(r.k), N.f3(r.LR_cv_auc), N.f3(r.LR_test_auc),
       N.f3(r.LGBM_kfeat_test_auc), N.f3(r.LGBM_1047_test_auc), f'{r.recovery_pct:.0f}']
      for r in N.LG_PARS.itertuples(index=False)],
     'k is the number of most stable features retained, chosen by bootstrap selection frequency '
     'in the training partition. Recovery is (AUC_small - 0.5) / (AUC_full - 0.5).'),

 _et('eTable 8. Stability of the feature ranking across bootstrap resamples',
     ['Outcome', 'Resamples', 'Features ever in top 20', 'In top 20 in >=50% of resamples',
      'In top 20 in >=80% of resamples', 'Median selection frequency of the point-estimate top 20',
      'Top feature by SHAP'],
     [[r.outcome, int(r.n_resamples), int(r.n_features_ever_in_top20),
       int(r.n_features_in_top20_over_50pct), int(r.n_features_in_top20_over_80pct),
       N.f2(r.median_freq_of_pointestimate_top20),
       N.one(N.LG_SHAP, outcome=r.outcome)['top1_shap']]
      for r in N.LG_STAB.itertuples(index=False)],
     f'With {N.comma(N.N_FEATURES)} candidate features and {N.N_TRAIN_EVENTS} training events, '
     f'the ranking is unstable by construction; feature-level statements in this study describe '
     f'one fit rather than a property of the pediatric ECG.'),

 _et('eTable 9. Models refitted on transportable features only',
     ['Outcome', 'Transportable features', 'Internal, all features', 'Internal, transportable',
      'External non-shunt, all features', 'External non-shunt, transportable',
      'External ECG-manifesting, all features', 'External ECG-manifesting, transportable'],
     [[r.outcome, f'{int(r.n_transportable)}/{int(r.n_candidate)}',
       N.f3(r.internal_full), N.f3(r.internal_transportable),
       N.f3(r.external_nonshunt_reporting_full), N.f3(r.external_nonshunt_reporting_transportable),
       N.f3(r.external_manifesting_reporting_full), N.f3(r.external_manifesting_reporting_transportable)]
      for r in N.LG_TRANS.itertuples(index=False)],
     'A feature counted as transportable when its univariate ROC-AUC reached 0.55 in both the '
     'internal training partition and the external selection half. All external values are '
     'computed in the external reporting half, which took no part in selecting the features.'),

 _et('eTable 10. Sample size against the criteria of Riley et al.',
     ['Candidate parameters', 'Description', 'Required n', 'Required events', 'Available n',
      'Available events', 'Events per parameter', 'Criteria met'],
     [[int(r.candidate_parameters), r.description, N.comma(r.required_N),
       N.comma(r.required_events), N.comma(r.available_N), int(r.available_events),
       N.f2(r.events_per_parameter), 'Yes' if r.meets else 'No']
      for r in N.RILEY.itertuples(index=False)],
     'Stated against the number of candidate predictors actually offered to the models, not the '
     'number that survive in the narrative.'),

 _et('eTable 11. Out-of-fold probabilities by chamber status among children with structural disease',
     ['Chamber', 'Never involved, n', 'Past involvement, n', 'Current involvement, n',
      'Past involvement operated, n', 'Median p, never', 'Median p, past', 'Median p, current',
      'ROC-AUC past vs never (95% CI)', 'Spearman rho'],
     [[r.chamber, int(r.n_never), int(r.n_past), int(r.n_current), int(r.past_operated),
       N.f3(r.med_never), N.f3(r.med_past), N.f3(r.med_current),
       f'{N.f3(r.auc_past_vs_never)} ({N.f3(r.lo)}-{N.f3(r.hi)})', N.f3(r.rho)]
      for r in N.CHAMBER.itertuples(index=False)],
     'Past involvement was coded negative during derivation, so a high probability in these '
     'chambers counts as a false positive and makes chamber-level specificity conservative. '
     'Because almost all chambers with past involvement had been operated, these values are '
     'consistent with persistent electrical sequelae of surgery or of the lesion and do not '
     'demonstrate detection of sub-threshold remodeling.'),

 _et('eTable 12. Threshold range in which the primary model carries more net benefit than '
     'imaging every child or no child',
     ['Outcome', 'Internal held-out', 'External, all ICD-10 codes'],
     [[o,
       (lambda r: f'{N.f2(r[0])}-{N.f2(r[1])}' if r else 'none')(N.dca_useful(o, 'internal held-out')),
       (lambda r: f'{N.f2(r[0])}-{N.f2(r[1])}' if r else 'none')(N.dca_useful(o, 'external, all ICD+'))]
      for o in N.OUTCOMES],
     'Decision-curve analysis over threshold probabilities of 0.01 to 0.50. Outside the stated '
     'range a policy of imaging every child performs as well or better. The external window is '
     'the practical consequence of the miscalibration in eTable 4.'),
]

# every string the citation scanner must see
ALL_TEXT = ([t['title'] for t in TABLES] + [t['note'] for t in TABLES] + FIGURE_LEGENDS
            + [p for _, ps in EMETHODS for p in ps] + [h for h, _ in EMETHODS]
            + [e['title'] for e in ETABLES] + [e.get('note', '') for e in ETABLES])
