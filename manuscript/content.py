"""Manuscript text for Pediatric Cardiology. LightGBM is the primary model.

Adapted from Manuscript/CHILDHEART_RF_Manuscript_source.md (v5/v6, random-forest primary).
The scientific content, the cohort description and the limitations are the authors' text; what
changed is (a) the primary estimator, because the selection rule stated in Methods picks
LightGBM, (b) the journal's required structure (Material and methods; Statements and
Declarations; no Key Points), and (c) every quantity, which is now interpolated from nums.py
rather than written out.

Citation syntax [[key]] is resolved by build_all.py against zot.py KEYS.
"""
import nums as N

GH_USER, GH_REPO = 'josenalencar', 'childheart-ecg-ml'

TITLE = ("Machine Learning Detects Pediatric Structural Heart Disease From the 12-Lead "
         "Electrocardiogram: External Validation in 10,352 Children")

# Author list confirmed by the corresponding author on 2026-10-13 (the ACC.27 abstract list).
# Degrees and ORCIDs are deliberately left as markers for the authors to complete in Word.
AUTHORS = ("José Nunes de Alencar Neto, MD; Thales Fernandes, [degree]; "
           "Claudionor Antônio dos Santos Filho, [degree]; "
           "Daniel Teixeira de Campos e Barreiro, [degree]; João Vitor Gardelli, [degree]; "
           "Nathalia Rego, [degree]; Julia Teuber Furtado, [degree]; "
           "Mariana Fuziy Nogueira De Marchi, [degree]; for the CHILDHEART Project")

AFFILIATIONS = [
    "Instituto Dante Pazzanese de Cardiologia, São Paulo, Brazil (Alencar Neto, Rego, Teuber Furtado, De Marchi)",
    "Centro Universitário São Camilo, São Paulo, Brazil (Fernandes)",
    "Universidade Potiguar, Natal, Brazil (dos Santos Filho)",
    "Faculdade de Medicina de Marília, Marília, Brazil (Teixeira de Campos e Barreiro)",
    "Universidade Santo Amaro, São Paulo, Brazil (Gardelli)",
    "ORCID: [TO ADD]",
]

CORRESPONDENCE = ("Corresponding author: José Nunes de Alencar Neto, MD, Instituto Dante Pazzanese "
                  "de Cardiologia, Avenida Dr Dante Pazzanese 500, Vila Mariana, São Paulo "
                  "04012-091, Brazil. E-mail: josenunesalencar@gmail.com")

KEYWORDS = ["Pediatric electrocardiography", "Congenital heart disease", "Machine learning",
            "External validation", "Diagnostic accuracy", "Reference standard"]

# ------------------------------------------------------------------ abstract (150-250 words)
_t = N.test_auc('ANY disease')
_e = N.ext_row('ANY disease')
_m = N.mani_row('ANY disease')
_mrf = N.mani_row('ANY disease', N.RF_TUNED)
_mxg = N.mani_row('ANY disease', N.XGB)
_lo_mani, _hi_mani = N.ranges(lambda o: N.mani_row(o)['manifesting_auc'])
_gap_lg = float(N.ext_row('ANY disease')['internal_minus_external_all'])
_gap_rf = float(N.ext_row('ANY disease', N.RF_TUNED)['internal_minus_external_all'])
_gap_xg = float(N.ext_row('ANY disease', N.XGB)['internal_minus_external_all'])

ABSTRACT = (
    f"Echocardiography is often unavailable where a child is first seen, whereas the 12-lead "
    f"electrocardiogram (ECG) is almost ubiquitous. Machine learning detects structural heart "
    f"disease from the pediatric ECG but loses accuracy on external validation. We hypothesized "
    f"that this loss reflects the reference standard against which the models are judged rather "
    f"than the capacity of the model, and that a materially stronger algorithm would therefore "
    f"not close it. Models were derived in {N.comma(N.N_INTERNAL)} Brazilian children "
    f"({N.N_CASES} with structural disease adjudicated by three pediatric cardiologists, "
    f"{N.comma(N.N_CONTROLS)} without clinical heart disease), using "
    f"{N.comma(N.N_FEATURES)} automatically extracted ECG features normalized for age. "
    f"Three estimator families were tuned by nested cross-validation within the training "
    f"partition; LightGBM was selected as primary and applied frozen to "
    f"{N.comma(N.N_EXT)} Chinese children whose diagnoses came from ICD-10 codes. "
    f"Held-out discrimination for any structural disease was {N.f3(_t['test_auc'])} "
    f"(95% CI, {N.f3(_t['ci_lo'])}–{N.f3(_t['ci_hi'])}), falling to "
    f"{N.f3(_e['external_all_icd_auc'])} against all ICD codes. The internal-to-external "
    f"difference was {N.f3(_gap_rf)}, {N.f3(_gap_xg)} and {N.f3(_gap_lg)} for the random "
    f"forest, XGBoost and LightGBM respectively: a stronger model raised accuracy at both ends "
    f"and closed none of the gap. Restricted to lesions expected to alter the ECG, external "
    f"discrimination recovered to {N.f3(_m['manifesting_auc'])} "
    f"({_lo_mani}–{_hi_mani} across chambers) and the choice of model ceased to matter. "
    f"Automated ECG analysis detects adjudicated pediatric structural heart disease; its "
    f"apparent external ceiling is set by what a diagnostic code means."
)

# ------------------------------------------------------------------ main text
def _delong_pair(outcome, a, b):
    r, flipped = N.delong(outcome, a, b)
    return float(r['delong_p'])

_p_lg_rf = _delong_pair('ANY disease', N.PRIMARY, N.RF_TUNED)
_p_lg_xg = _delong_pair('ANY disease', N.PRIMARY, N.XGB)
_p_rf_rf = _delong_pair('ANY disease', N.RF_SHIPPED, N.RF_TUNED)

_cal_int = {o: N.cal_row(o, 'internal held-out') for o in N.OUTCOMES}
_cal_ext = {o: N.cal_row(o, 'external, all ICD+') for o in N.OUTCOMES}
_ns_any = N.cal_row('ANY disease', 'external, non-shunt')
_riley_full = N.riley(N.N_FEATURES)
_riley_small = N.riley(4)
_stab_any = N.stab('ANY disease')
_lot_any = N.lot('ANY disease')
_you_any = N.ops('ANY disease', 'Youden (OOF train)')
_s95_any = N.ops('ANY disease', '>=95% specificity (OOF train)')

MAIN = [
 ("Introduction", [
  f"Structural heart disease in children encompasses complex congenital defects, "
  f"cardiomyopathies and acquired valvular lesions that require prompt diagnosis to prevent "
  f"irreversible pulmonary vascular change, ventricular dysfunction and premature "
  f"death.[[liu2019]] Transthoracic echocardiography provides definitive anatomical and "
  f"functional assessment, but its availability is concentrated in tertiary centers; in "
  f"community hospitals and primary triage clinics, access to pediatric echocardiography is "
  f"constrained by workforce deficits and the capital cost of equipment.",

  f"The 12-lead ECG is ubiquitous, inexpensive and routinely obtained during pediatric "
  f"evaluation, yet manual interpretation of pediatric tracings remains difficult. The normal "
  f"pediatric ECG changes across infancy and childhood, from the physiological right "
  f"ventricular dominance of the neonate toward left ventricular predominance with somatic "
  f"maturation.[[davignon]] These wide, age-dependent reference intervals suppress human "
  f"sensitivity for early chamber enlargement and for congenital anomalies,[[evans]] and "
  f"age-specific milestones are needed before a pediatric tracing can be read "
  f"quickly.[[milestones]]",

  f"Machine learning has proved useful in automated electrocardiographic "
  f"analysis,[[alzaiti, attia]] but the pediatric literature carries recurring methodological "
  f"limitations. Most algorithms are derived in single-center cohorts without external "
  f"validation,[[mori, lee2025, pachiyannan, devries]] and the studies with large sample sizes "
  f"frequently rely on administrative billing codes as surrogate ground truth rather than on "
  f"verified imaging.[[khan]] An administrative code designates a diagnostic category, such as "
  f"a small atrial or ventricular septal defect, without establishing whether secondary "
  f"hemodynamic chamber stress or any electrophysiological consequence is present. A model "
  f"judged against such a label is therefore penalized for being right about physiology.",

  f"We hypothesized that the loss of accuracy seen when pediatric ECG models are transported "
  f"to an external cohort is governed by the reference standard rather than by the capacity of "
  f"the model, and that a materially stronger learning algorithm would raise accuracy in both "
  f"cohorts while closing none of the gap between them. To test this we derived automated "
  f"models against expert multimodality adjudication of chamber involvement in a Brazilian "
  f"pediatric cohort, compared three estimator families under identical tuning, and applied "
  f"the selected model unchanged to {N.comma(N.N_EXT)} Chinese children labelled by ICD-10 "
  f"codes, varying only the external case definition."]),

 ("Material and methods", [
  "##Study design and reporting",
  f"This was a diagnostic prediction-model development and external validation study using "
  f"existing cohort and health-record data. Reporting follows the TRIPOD+AI "
  f"statement[[tripodai, tripod]] and STARD 2015,[[stard]] and risk of bias was appraised with "
  f"PROBAST.[[probast]] Because cases and controls were sampled separately and from different "
  f"care settings, the PROBAST participant domain carries a high risk of bias, which tends to "
  f"inflate internal discrimination; the analysis domain is at lower risk, with a held-out test "
  f"set and frozen external application. The study was not prospectively registered and no "
  f"protocol was published; the analysis plan was fixed before the held-out test set was "
  f"opened, but was not deposited in advance.",

  "##Cohorts and reference standards",
  f"The derivation cohort (CHILDHEART) comprised two groups recruited separately, as in the "
  f"parent CHILDHEART studies.[[milestones]] Cases were 376 consecutive children aged 0 to 14 "
  f"years admitted to the pediatric cardiology ward or intensive care unit of Instituto Dante "
  f"Pazzanese de Cardiologia, São Paulo, Brazil, during 2023; 27 were excluded (isolated "
  f"arrhythmia without structural disease, ventricular pacing, or an unavailable, illegible or "
  f"artifact-compromised tracing), leaving {N.N_CASES}. Controls were "
  f"{N.comma(N.N_CONTROLS)} children who had a routine ECG for non-cardiac reasons, such as "
  f"preoperative assessment or sports clearance, in tele-electrocardiography services serving "
  f"general care; they were selected for the absence of any clinical indication of heart "
  f"disease rather than for a normal ECG, at a targeted case-to-control ratio of 1:4, and did "
  f"not undergo echocardiography. One 12-lead ECG per child was analyzed "
  f"(n = {N.comma(N.N_INTERNAL)}), so cases made up {N.pct(N.CASE_FRACTION)}% of the cohort by "
  f"design rather than by observed prevalence.",

  f"The reference standard was the determination, made independently by three pediatric "
  f"cardiologists blinded to the ECG, of enlargement, hypertrophy or disease in each chamber "
  f"(right atrium, left atrium, right ventricle, left ventricle), integrating history, physical "
  f"examination, echocardiography, chest radiography, cardiac magnetic resonance where "
  f"available, and perioperative findings. Echocardiography was the principal but not the sole "
  f"source, and the determination distinguished current from past (resolved) involvement. A "
  f"fixed stratified 80/20 train/test split on overall disease status gave "
  f"{N.comma(N.N_TRAIN)} training children ({N.N_TRAIN_EVENTS} with disease) and {N.N_TEST} "
  f"held-out children ({N.N_TEST_EVENTS} with disease). This division was identical across all "
  f"models.",

  f"External validation used the Zhengzhou University pediatric ECG database from "
  f"China,[[zzu]] comprising 14,190 tracings from 11,643 patients, in which disease was "
  f"classified through an ICD-10 crosswalk rather than by imaging. The cohort was restricted to "
  f"complete 12-lead recordings, excluding 1,856 nine-lead tracings that lacked leads V2, V4 "
  f"and V6, and to one primary tracing per child, excluding 1,979 repeat recordings; after 3 "
  f"further exclusions for signal-extraction failure, {N.comma(N.N_EXT)} records from "
  f"{N.comma(N.N_EXT)} children remained, of which {N.comma(N.N_EXT_EVENTS)} carried a cardiac "
  f"ICD-10 code (Fig. 1).",

  "##Electrocardiographic feature extraction",
  f"Features were extracted automatically with the ecg-vcg-viewer pipeline (delineator D2, "
  f"extractor E3, measurement contract M2, commit {N.ENGINE_COMMIT}, feature tables v4_E3). No "
  f"manual measurement entered the analytical stream. The extractor produced 1,107 raw "
  f"parameters per tracing. Biologically impossible values were set to missing and imputed with "
  f"training-set medians, using thresholds of heart rate outside 25 to 320 beats/min, PR "
  f"interval outside 50 to 400 ms, QRS duration outside 30 to 250 ms, QT outside 150 to 800 ms, "
  f"QTc outside 200 to 700 ms, P-wave duration outside 10 to 220 ms, and absolute amplitudes "
  f"above 8 mV. After filtering, {N.comma(N.N_FEATURES)} features were retained. They comprised "
  f"interval durations, lead-level amplitudes and areas, planar vectorcardiographic indices "
  f"(spatial and mean QRS-T angles[[bergfeldt]] and total cosine R-to-T[[nussinovitch]]), "
  f"Cornell[[casale]] and Peguero-Lo Presti[[peguero]] criteria, Romhilt-Estes "
  f"scores[[romhilt]] and Minnesota ST/T classifications.[[prineas]]",

  f"Continuous features were converted to age-specific z scores across eight chronological "
  f"partitions spanning 0 to 300 months. Normalization constants were estimated exclusively "
  f"among disease-free children within the training fold, linearly interpolated between "
  f"partition midpoints and truncated at ±8 standard deviations. Binary indicators, ordinal "
  f"variables, counts and age in months were kept raw. The fitted normalization matrix was "
  f"applied to the external cohort without re-estimation.",

  "##Model development and selection of the primary estimator",
  f"Five one-versus-rest targets were modelled: any structural heart disease and disease of "
  f"each of the four chambers. For a chamber target, the comparison group included every child "
  f"without disease in that chamber, including children with disease confined to another "
  f"chamber.",

  f"Three estimator families were fitted to the identical training partition, feature matrix "
  f"and seed: a random forest, gradient-boosted trees (XGBoost[[xgboost]]) and gradient-boosted "
  f"trees with leaf-wise growth (LightGBM[[lightgbm]]). Hyperparameters were selected by "
  f"randomized search (15 draws, three-fold inner cross-validation) inside a five-fold outer "
  f"loop, so the cross-validated estimate carries no tuning optimism. The random forest was "
  f"additionally fitted with the hyperparameters used in the parent CHILDHEART analyses (600 "
  f"trees, balanced class weights, minimum leaf size 2, seed 42), to separate an estimator "
  f"effect from an untuned baseline.",

  f"The primary estimator was chosen using training data only, by the highest nested "
  f"cross-validated area under the receiver operating characteristic curve (ROC-AUC) for the "
  f"primary outcome of any structural disease: {N.f3(N.nested('ANY disease'))} for LightGBM, "
  f"{N.f3(N.nested('ANY disease', N.XGB))} for XGBoost and "
  f"{N.f3(N.nested('ANY disease', N.RF_TUNED))} for the tuned random forest. LightGBM was "
  f"therefore adopted as the primary model for all five outcomes, and the random forest and "
  f"XGBoost are reported throughout as comparators. The selected configuration was refitted on "
  f"the whole training partition, applied once to the held-out test set, and applied unchanged "
  f"to the external cohort. No external data informed model selection, tuning or thresholds.",

  f"Decision thresholds were fixed on out-of-fold predictions within the training partition "
  f"only, at the point maximizing Youden's J statistic and at the lowest threshold achieving at "
  f"least 95% specificity, and then transferred unaltered to the held-out test set.",

  "##External case definitions",
  f"Three pre-specified external case definitions were applied to the same frozen models, "
  f"each against disease-free children. The first counted every child carrying a cardiac "
  f"ICD-10 code. The second excluded children whose only coded lesion was an isolated atrial "
  f"septal defect, ventricular septal defect or patent ductus arteriosus, giving a non-shunt, "
  f"CHILDHEART-like spectrum. The third restricted cases to lesions expected on physiological "
  f"grounds to alter the ECG, using positive lesion lists fixed before analysis: complex "
  f"congenital heart disease, pulmonary hypertension and tetralogy of Fallot for any disease; "
  f"cardiomyopathies, aortic stenosis and coarctation for the left ventricle; tetralogy of "
  f"Fallot, pulmonary stenosis and pulmonary hypertension for the right ventricle; tricuspid "
  f"disease, pulmonary hypertension and complex disease for the right atrium; and mitral "
  f"disease, atrioventricular septal defect and complex disease for the left atrium.",

  "##Chamber status among children with structural disease",
  f"For each chamber of the {N.N_CASES} children with structural disease, the adjudication "
  f"recorded whether involvement was current, past (resolved, most often after surgical repair) "
  f"or absent. Only current involvement was coded positive during derivation; past and absent "
  f"involvement were both coded negative. Out-of-fold probabilities from five-fold "
  f"cross-validation over the whole cohort were compared across the three statuses, to ask "
  f"whether chambers the models were trained to call negative nonetheless carried an "
  f"electrocardiographic signal.",

  "##Feature transportability",
  f"To ask whether the internal-to-external decline reflected features that do not carry "
  f"across cohorts rather than labels alone, a direction-agnostic univariate ROC-AUC was "
  f"computed for every feature in the internal training partition and in the external cohort. "
  f"Choosing features with external data and then reporting external performance in the same "
  f"children would overstate transportability, so the external cohort was split once, at "
  f"patient level and stratified by disease status, into a selection half and a reporting half. "
  f"A feature was transportable when its univariate ROC-AUC reached at least 0.55 in both the "
  f"internal training partition and the external selection half; models were refitted on each "
  f"outcome's transportable subset, and every external estimate for them was computed in the "
  f"reporting half only.",

  "##Feature importance, stability and a parsimony floor",
  f"Importance was ranked by TreeSHAP,[[treeshap]] with the model's own gain-based importance "
  f"as a cross-check, because impurity and gain measures are biased toward continuous and "
  f"high-cardinality predictors, which describes nearly all "
  f"{N.comma(N.N_FEATURES)} features here. Ranking stability was estimated by refitting on "
  f"{int(_stab_any['n_resamples'])} bootstrap resamples of the training partition and recording "
  f"how often each feature re-entered the top twenty. As a parsimony floor, penalized logistic "
  f"regression and LightGBM restricted to the five, ten and twenty most stable features were "
  f"compared with the full model. Sample size was assessed against the criteria of Riley "
  f"et al.[[riley]] for the number of candidate predictors actually offered to the models, not "
  f"the number surviving in the narrative.",

  "##Statistical analysis",
  f"Discrimination was summarized by ROC-AUC and by the area under the precision-recall curve, "
  f"with 2,000-sample bootstrap 95% confidence intervals. Competing models were compared with "
  f"DeLong's test for correlated ROC curves.[[delong]] Calibration was assessed by "
  f"calibration-in-the-large and the calibration slope, obtained by regressing the outcome on "
  f"the linear predictor with and without that predictor as a fixed offset, alongside the Brier "
  f"score. Clinical usefulness was assessed by decision-curve analysis.[[vickers]] Because the "
  f"held-out estimate is a single draw, the outcome-stratified split was repeated "
  f"{int(_lot_any['n_splits'])} times and the distribution of held-out discrimination reported. "
  f"Trends across chamber status were assessed with Spearman's rank correlation and differences "
  f"between two statuses with one-sided Mann-Whitney U tests. Analyses used Python 3.13 with "
  f"scikit-learn, XGBoost and LightGBM."]),
]

# ------------------------------------------------------------------ results
def _auc_list(fn, fmt=N.f3, sep=', ', last=' and '):
    """'0.922 for the right atrium, 0.927 for the left atrium ...' built from the files."""
    parts = [f"{fmt(fn(o))} for the {N.SHORT[o].replace(' disease','')}" for o in N.OUTCOMES[1:]]
    return sep.join(parts[:-1]) + last + parts[-1]

_cham = {c: N.chamber(c) for c in ['Right ventricle', 'Right atrium', 'Left atrium', 'Left ventricle']}
_tr_any = N.trans('ANY disease')
_lo_pars, _hi_pars = N.ranges(lambda o: N.pars(o, 10)['recovery_pct'], fmt=lambda x: f'{x:.0f}')
_dca_int = N.dca_useful('ANY disease', 'internal held-out')
_dca_ext = N.dca_useful('ANY disease', 'external, all ICD+')

RESULTS = [
  "##Cohorts",
  f"The derivation cohort included {N.comma(N.N_INTERNAL)} children (median age 51 months; "
  f"IQR 15–105; 39.0% aged under 3 years; 50.0% male; cases {N.pct(N.CASE_FRACTION)}% by "
  f"design). The external cohort was older (median age 114 months; IQR 69–153; 9.2% aged under "
  f"3 years; 57.7% male), and {N.comma(N.N_EXT_EVENTS)} of {N.comma(N.N_EXT)} children "
  f"({N.pct(N.N_EXT_EVENTS / N.N_EXT)}%) carried a cardiac ICD-10 code. Data cleaning nulled "
  f"1,344 physiologically implausible entries across 25 features internally, mostly lead-level "
  f"T-wave intrinsicoid deflections, and 8,021 values across 165 features externally. "
  f"Automated measurements agreed closely with manual calipers for ventricular depolarization "
  f"(Pearson r = 0.977 for the R or R′ wave in V1, 0.973 for R in V6, 0.949 for S in V4 and "
  f"0.947 for R in lead I), but agreement for P-wave duration in lead II was poor "
  f"(r = 0.281; mean automated 74.1 ms versus 68.7 ms by caliper; bias +5.44 ms).",

  "##Internal discrimination of the primary model",
  f"On the held-out test set ({N.N_TEST} children, {N.N_TEST_EVENTS} with structural disease), "
  f"LightGBM reached a ROC-AUC of {N.f3(_t['test_auc'])} "
  f"(95% CI, {N.f3(_t['ci_lo'])}–{N.f3(_t['ci_hi'])}) for any structural disease, with a "
  f"precision-recall AUC of {N.f3(_you_any['pr_auc'])} against a case fraction of "
  f"{N.f3(_you_any['prevalence'])} and a Brier score of {N.f3(_you_any['brier'])}. "
  f"Chamber-level discrimination was {_auc_list(lambda o: N.test_auc(o)['test_auc'])}.",

  f"At the Youden threshold fixed on out-of-fold training predictions "
  f"({N.f3(_you_any['threshold'])}), the any-disease model reached a sensitivity of "
  f"{N.pct(_you_any['sensitivity'])}% and a specificity of {N.pct(_you_any['specificity'])}%, "
  f"with a positive likelihood ratio of {N.f2(_you_any['lr_pos'])} and a negative predictive "
  f"value of {N.pct(_you_any['npv'])}%. Moving the threshold to the point giving at least 95% "
  f"specificity on the training folds ({N.f3(_s95_any['threshold'])}) gave a specificity of "
  f"{N.pct(_s95_any['specificity'])}% and a positive likelihood ratio of "
  f"{N.f2(_s95_any['lr_pos'])} at a sensitivity of {N.pct(_s95_any['sensitivity'])}%. Because "
  f"the case fraction was fixed by design, predictive values do not transfer to clinical "
  f"populations; likelihood ratios are the transportable quantities.",

  "##Estimator comparison and the invariance of the cross-cohort gap",
  f"The three estimator families separated internally. For any structural disease the held-out "
  f"ROC-AUC was {N.f3(N.test_auc('ANY disease', N.RF_SHIPPED)['test_auc'])} for the random "
  f"forest as used in the parent analyses, "
  f"{N.f3(N.test_auc('ANY disease', N.RF_TUNED)['test_auc'])} for the tuned forest, "
  f"{N.f3(N.test_auc('ANY disease', N.XGB)['test_auc'])} for XGBoost and "
  f"{N.f3(_t['test_auc'])} for LightGBM. LightGBM exceeded the tuned forest on DeLong's test "
  f"({N.pval(_p_lg_rf)}) but was indistinguishable from XGBoost ({N.pval(_p_lg_xg)}). Tuning "
  f"the forest itself changed nothing ({N.pval(_p_rf_rf)}), so the difference is attributable "
  f"to the estimator rather than to an untuned baseline. Across the four chamber outcomes no "
  f"pairwise difference reached significance.",

  f"Applied unchanged to the external cohort against all ICD-coded cases, discrimination for "
  f"any structural disease fell to {N.f3(_e['external_all_icd_auc'])} "
  f"(95% CI, {N.f3(_e['ext_all_ci_lo'])}–{N.f3(_e['ext_all_ci_hi'])}). The decisive result is "
  f"that the internal-to-external difference did not depend on the estimator: it was "
  f"{N.f3(_gap_rf)} for the random forest, {N.f3(_gap_xg)} for XGBoost and {N.f3(_gap_lg)} for "
  f"LightGBM. A materially stronger algorithm raised discrimination at both ends and closed "
  f"none of the gap between them.",

  "##External case definition",
  f"Varying the external case definition, with the models untouched, moved discrimination far "
  f"more than changing the algorithm had. Excluding children whose only lesion was an isolated "
  f"shunt raised any-disease discrimination from {N.f3(_e['external_all_icd_auc'])} to "
  f"{N.f3(_e['external_nonshunt_auc'])} "
  f"(95% CI, {N.f3(_e['ext_ns_ci_lo'])}–{N.f3(_e['ext_ns_ci_hi'])}). Restricting cases to "
  f"lesions expected to alter the ECG — complex congenital disease, pulmonary hypertension and "
  f"tetralogy of Fallot, {int(_m['n_manifesting'])} children against "
  f"{N.comma(int(_m['n_disease_free']))} disease-free children — recovered a ROC-AUC of "
  f"{N.f3(_m['manifesting_auc'])} (95% CI, {N.f3(_m['ci_lo'])}–{N.f3(_m['ci_hi'])}). The "
  f"chamber models recovered in the same way, to "
  f"{_auc_list(lambda o: N.mani_row(o)['manifesting_auc'])}.",

  f"Under this definition the estimators became indistinguishable. For any structural disease "
  f"the random forest reached {N.f3(_mrf['manifesting_auc'])}, XGBoost "
  f"{N.f3(_mxg['manifesting_auc'])} and LightGBM {N.f3(_m['manifesting_auc'])}, so the "
  f"advantage LightGBM held internally shrank from "
  f"{N.f3(float(_t['test_auc']) - float(N.test_auc('ANY disease', N.RF_TUNED)['test_auc']))} to "
  f"{N.f3(float(_m['manifesting_auc']) - float(_mrf['manifesting_auc']))} once the label "
  f"described physiology rather than anatomy. Cleaning the reference standard removed the "
  f"apparent benefit of the stronger model.",

  "##Never, past and current chamber involvement",
  f"Among the {N.N_CASES} children with structural disease, 148 chambers in 102 children had "
  f"past involvement, {sum(int(_cham[c]['past_operated']) for c in _cham)} of them after "
  f"cardiac surgery. Although past and absent involvement were both coded negative during "
  f"derivation, out-of-fold probabilities were higher for chambers with past involvement than "
  f"for never-involved chambers in all four models (ROC-AUC "
  f"{N.f3(min(_cham[c]['auc_past_vs_never'] for c in _cham))} to "
  f"{N.f3(max(_cham[c]['auc_past_vs_never'] for c in _cham))}). For the ventricles, current "
  f"involvement scored above past involvement; for the atria it did not, past involvement "
  f"scoring as high as current involvement in the right atrium "
  f"({N.f3(_cham['Right atrium']['med_past'])} versus "
  f"{N.f3(_cham['Right atrium']['med_current'])}) and higher in the left atrium "
  f"({N.f3(_cham['Left atrium']['med_past'])} versus "
  f"{N.f3(_cham['Left atrium']['med_current'])}). Because almost every chamber with past "
  f"involvement had been treated surgically, these probabilities are consistent with electrical "
  f"sequelae of surgery, or of the lesion, persisting after the chamber load has been relieved; "
  f"they do not demonstrate detection of sub-threshold remodeling, and the design cannot "
  f"separate the two explanations. These 148 chamber-level instances were controls during "
  f"training, so high predictions in them count as false positives, making chamber-level "
  f"specificity conservative.",

  "##Calibration and clinical usefulness",
  f"Discrimination transported; calibrated probability did not. On the held-out test set "
  f"calibration slopes exceeded unity for every outcome "
  f"({N.f2(min(_cal_int[o]['calibration_slope'] for o in N.OUTCOMES))} to "
  f"{N.f2(max(_cal_int[o]['calibration_slope'] for o in N.OUTCOMES))}), indicating predicted "
  f"probabilities compressed toward the fixed case fraction. Externally the pattern inverted: "
  f"slopes fell below unity for every outcome "
  f"({N.f2(min(_cal_ext[o]['calibration_slope'] for o in N.OUTCOMES))} to "
  f"{N.f2(max(_cal_ext[o]['calibration_slope'] for o in N.OUTCOMES))}) and "
  f"calibration-in-the-large was negative throughout, indicating systematic over-prediction of "
  f"risk. Over-prediction was larger still against the non-shunt definition, where mean "
  f"predicted risk was {N.f2(_ns_any['mean_predicted'])} against an observed prevalence of "
  f"{N.f3(_ns_any['observed_prevalence'])} for any structural disease "
  f"(calibration-in-the-large {N.f2(_ns_any['calibration_in_the_large'])}). Brier scores, "
  f"reported alone in earlier analyses, do not reveal this. Any deployment would require "
  f"recalibration in the target population.",

  f"Decision-curve analysis gave the same message in the units a clinician acts on. Internally "
  f"the model carried higher net benefit than imaging every child or no child across "
  f"essentially the whole plausible threshold range "
  f"({N.f2(_dca_int[0])} to {N.f2(_dca_int[1])}). Externally the window narrowed to "
  f"{N.f2(_dca_ext[0])} to {N.f2(_dca_ext[1])} for any structural disease; outside it, imaging "
  f"every child performs as well or better. The clinically useful operating range is therefore "
  f"narrow and prevalence-dependent, which is the practical consequence of the miscalibration "
  f"above.",

  "##Sample size, ranking stability and a parsimony floor",
  f"By the criteria of Riley et al., a model with four binary items at the observed case "
  f"fraction would require {N.comma(_riley_small['required_N'])} participants, and the training "
  f"partition provided {N.comma(N.N_TRAIN)} with {N.N_TRAIN_EVENTS} events. The "
  f"{N.comma(N.N_FEATURES)} candidate features actually offered to the models would require "
  f"{N.comma(_riley_full['required_N'])} participants and "
  f"{N.comma(_riley_full['required_events'])} events, roughly "
  f"{int(round(float(_riley_full['required_N']) / N.N_TRAIN))} times what was available, giving "
  f"{N.f2(_riley_full['events_per_parameter'])} events per candidate predictor. The models were "
  f"therefore fitted far outside the regime in which a feature ranking is stable, and the "
  f"bootstrap confirms it: across resamples of the training partition, "
  f"{int(_stab_any['n_features_ever_in_top20'])} distinct features entered the top twenty for "
  f"any disease at least once, and only "
  f"{int(_stab_any['n_features_in_top20_over_80pct'])} appeared there in at least 80% of "
  f"resamples. Feature-level claims in this study should be read as descriptions of one fit "
  f"rather than as properties of the pediatric ECG.",

  f"Restricting the model to the most stable features cost little. For any structural disease, "
  f"penalized logistic regression on ten features reached "
  f"{N.f3(N.pars('ANY disease', 10)['LR_test_auc'])} against {N.f3(_t['test_auc'])} for "
  f"LightGBM on all {N.comma(N.N_FEATURES)}, recovering "
  f"{N.f2(N.pars('ANY disease', 10)['recovery_pct'])}% of the chance-corrected discrimination "
  f"of the full model; across the five outcomes the ten-feature logistic models recovered "
  f"{_lo_pars}% to {_hi_pars}%. Most of what a thousand features buy is therefore available "
  f"from a handful, which matters in a setting where the extraction pipeline, not the ECG, is "
  f"the barrier to use.",

  "##Which measurements the primary model uses",
  f"Under the primary model the ranking did not reproduce the one reported for the random "
  f"forest in earlier versions of this analysis, in which the mean QRS-T angle led every "
  f"classifier. Ranked by TreeSHAP, the leading feature was "
  f"{N.one(N.LG_SHAP, outcome='ANY disease')['top1_shap']} for any structural disease, "
  f"{N.one(N.LG_SHAP, outcome='Right atrial')['top1_shap']} for the right atrium, "
  f"{N.one(N.LG_SHAP, outcome='Left atrial')['top1_shap']} for the left atrium, "
  f"{N.one(N.LG_SHAP, outcome='Right ventricular')['top1_shap']} for the right ventricle and "
  f"{N.one(N.LG_SHAP, outcome='Left ventricular')['top1_shap']} for the left ventricle, so the "
  f"mean QRS-T angle led only one of the five. The model's own gain-based ranking disagreed "
  f"with TreeSHAP about the leading feature in "
  f"{sum(1 for o in N.OUTCOMES if N.one(N.LG_SHAP, outcome=o)['top1_shap'] != N.one(N.LG_SHAP, outcome=o)['top1_gain'])} "
  f"of the five outcomes, sharing "
  f"{int(N.one(N.LG_SHAP, outcome='ANY disease')['overlap_top10'])} of its top ten for any "
  f"disease, which is the behaviour expected of impurity-type measures when nearly every "
  f"candidate is continuous.",

  f"These rankings should not be over-read. Across bootstrap resamples of the training "
  f"partition, {int(_stab_any['n_features_ever_in_top20'])} distinct features entered the top "
  f"twenty for any disease at least once and only "
  f"{int(_stab_any['n_features_in_top20_over_80pct'])} did so in at least 80% of resamples; the "
  f"corresponding counts for the chamber outcomes were "
  f"{N.comma(min(int(N.stab(o)['n_features_ever_in_top20']) for o in N.OUTCOMES[1:]))} to "
  f"{N.comma(max(int(N.stab(o)['n_features_ever_in_top20']) for o in N.OUTCOMES[1:]))} and "
  f"{min(int(N.stab(o)['n_features_in_top20_over_80pct']) for o in N.OUTCOMES[1:])} to "
  f"{max(int(N.stab(o)['n_features_in_top20_over_80pct']) for o in N.OUTCOMES[1:])}. The "
  f"ranking is therefore unstable in both estimator families, and the specific features a given "
  f"fit elevates are not a property of the pediatric ECG. What survives resampling is the "
  f"coarser observation that QRS depolarization and atrial measurements, rather than "
  f"repolarization indices, are the ones that transport to the external cohort.",

  "##Variability of the held-out estimate and subgroups",
  f"The held-out estimate is one draw. Repeating the outcome-stratified split "
  f"{int(_lot_any['n_splits'])} times gave a mean ROC-AUC of {N.f3(_lot_any['mean'])} "
  f"(SD {N.f3(_lot_any['sd'])}; range {N.f3(_lot_any['min'])} to {N.f3(_lot_any['max'])}) for "
  f"any structural disease, placing the reported value at about the "
  f"{int(_lot_any['percentile_of_reported'])}th percentile of its own distribution. The "
  f"reported figure for the primary outcome is therefore a conservative draw rather than a "
  f"favourable one, as it is for the right atrium and right ventricle; for the left ventricle "
  f"the fixed split sits above the median instead (eTable 6). This comparison reuses "
  f"hyperparameters tuned on the original training partition and is to that extent mildly "
  f"optimistic about the distribution itself.",

  f"Discrimination for any structural disease was similar in girls "
  f"({N.f3(N.sub('ANY disease', 'female')['auc'])}) and boys "
  f"({N.f3(N.sub('ANY disease', 'male')['auc'])}), and across age strata "
  f"({N.f3(N.sub('ANY disease', 'age <3y')['auc'])} under 3 years, "
  f"{N.f3(N.sub('ANY disease', 'age 3-8y')['auc'])} from 3 to 8 years and "
  f"{N.f3(N.sub('ANY disease', 'age >=8y')['auc'])} at 8 years and over). Event counts within "
  f"strata are small and these comparisons are descriptive. Ethnicity was not recorded, so "
  f"fairness across ethnic groups could not be assessed.",

  "##Feature transportability",
  f"Loss of discrimination between cohorts was concentrated in repolarization features. The "
  f"vectorcardiographic indices that led the importance ranking internally showed the largest "
  f"internal-to-external declines of all {N.comma(N.N_FEATURES)} features, while QRS "
  f"depolarization and atrial measures retained theirs. Of the "
  f"{N.comma(N.N_FEATURES)} candidates, {int(_tr_any['n_transportable'])} met the "
  f"transportability criterion for any disease. Refitting on that subset changed internal "
  f"held-out discrimination from {N.f3(_tr_any['internal_full'])} to "
  f"{N.f3(_tr_any['internal_transportable'])} and external discrimination in the reporting half "
  f"from {N.f3(_tr_any['external_nonshunt_reporting_full'])} to "
  f"{N.f3(_tr_any['external_nonshunt_reporting_transportable'])} against non-shunt disease, and "
  f"from {N.f3(_tr_any['external_manifesting_reporting_full'])} to "
  f"{N.f3(_tr_any['external_manifesting_reporting_transportable'])} against ECG-manifesting "
  f"disease. Features that do not transport therefore contribute to the gap, but far less than "
  f"the case definition does.",

  "##Measurement engine sensitivity",
  f"Upgrading the feature extractor from E2 to E3, with cohorts, model architectures and seeds "
  f"held fixed, improved internal discrimination across all five classifiers (+0.007 to +0.015 "
  f"ROC-AUC) but left external discrimination against ICD-10 labels essentially unchanged "
  f"(−0.005 to +0.005). Better measurement improved classification against the adjudicated "
  f"reference standard and did nothing against administrative codes, which is the same "
  f"conclusion the estimator comparison reaches from the other direction.",
]

MAIN.append(("Results", RESULTS))

DISCUSSION = [
  f"Machine learning on automated 12-lead ECG measurements identified pediatric structural "
  f"heart disease adjudicated by expert review of echocardiographic, clinical and surgical "
  f"records, with a held-out ROC-AUC of {N.f3(_t['test_auc'])} for any structural disease. "
  f"Transported unchanged to an independent cohort labelled by diagnosis codes, discrimination "
  f"fell to {N.f3(_e['external_all_icd_auc'])}. The central finding of this study is what did "
  f"and did not move that gap.",

  f"Changing the algorithm did not move it. Three estimator families, tuned identically, "
  f"separated internally by up to "
  f"{N.f3(float(_t['test_auc']) - float(N.test_auc('ANY disease', N.RF_SHIPPED)['test_auc']))} "
  f"ROC-AUC, and the internal-to-external difference was {N.f3(_gap_rf)}, {N.f3(_gap_xg)} and "
  f"{N.f3(_gap_lg)} across them. Changing the reference standard moved it a great deal: with "
  f"the models untouched, restricting cases to lesions that load or injure the heart recovered "
  f"discrimination to {N.f3(_m['manifesting_auc'])}. The ceiling on external accuracy in this "
  f"setting is set by what the comparison diagnosis means, not by the capacity of the learner. "
  f"That the internal advantage of gradient boosting also collapsed under the cleaner label, "
  f"from {N.f3(float(_t['test_auc']) - float(N.test_auc('ANY disease', N.RF_TUNED)['test_auc']))} "
  f"to {N.f3(float(_m['manifesting_auc']) - float(_mrf['manifesting_auc']))}, suggests that part "
  f"of what the stronger model gained internally was fit to the idiosyncrasies of the label "
  f"rather than to cardiac physiology.",

  f"Administrative coding records anatomical presence, not secondary hemodynamic change. In "
  f"pediatric cohorts, codes for small atrial or ventricular septal communications often "
  f"describe restrictive lesions producing negligible chamber dilation, hypertrophy or "
  f"conduction delay.[[khan]] The models assigned low probabilities to those tracings, which is "
  f"correct physiologically and counts as a false negative against the code. This has a direct "
  f"implication for how pediatric ECG artificial intelligence should be evaluated: a study "
  f"reporting external ROC-AUC against ICD codes alone cannot distinguish a model that has "
  f"failed from a model that is being asked the wrong question, and should report at least one "
  f"physiology-anchored case definition alongside.",

  f"Features that do not transport play a smaller part. Vectorcardiographic repolarization "
  f"indices lost most of their discriminative value externally, whereas QRS depolarization and "
  f"atrial measures retained theirs; but refitting the primary model on the transportable "
  f"subset changed external discrimination by only "
  f"{N.f3(min(float(N.trans(o)['external_nonshunt_reporting_transportable']) - float(N.trans(o)['external_nonshunt_reporting_full']) for o in N.OUTCOMES))} "
  f"to "
  f"{N.f3(max(float(N.trans(o)['external_nonshunt_reporting_transportable']) - float(N.trans(o)['external_nonshunt_reporting_full']) for o in N.OUTCOMES))} "
  f"across the five outcomes, and for the primary outcome by "
  f"{N.f3(float(N.trans('ANY disease')['external_nonshunt_reporting_transportable']) - float(N.trans('ANY disease')['external_nonshunt_reporting_full']))}. "
  f"Discarding three quarters of the feature set therefore bought almost nothing, which places "
  f"the weight of the explanation on the label rather than on the features. Because "
  f"{N.PCT_POSTSURGICAL}% of "
  f"derivation cases had undergone cardiac surgery, a wide QRS-T angle may partly encode "
  f"postsurgical conduction change rather than chamber disease itself, which would account for "
  f"its weak external behaviour. We draw this conclusion from the univariate transportability "
  f"analysis rather than from the importance ranking, because the ranking proved unstable and "
  f"estimator-dependent: the mean QRS-T angle led every classifier under the random forest but "
  f"only one of five under the primary model. Mechanistic claims built on feature importance in "
  f"a study of this size should be treated as hypotheses.",

  f"The chamber-status analysis argues for caution rather than for sub-threshold detection. "
  f"Chambers whose involvement had resolved, nearly all after surgery, and which the models "
  f"were trained to call negative, nonetheless received intermediate or high probabilities, and "
  f"in the atria they scored as high as chambers with current involvement. Together with the "
  f"dominance of the QRS-T angle, this suggests that part of what the models detect is the "
  f"electrocardiographic trace of repair rather than current chamber load, which is why "
  f"validation in unoperated children is the necessary next step.",

  f"Two findings bear on deployment. First, discrimination transported but calibration did "
  f"not: predicted probabilities were systematically too high externally, and decision-curve "
  f"analysis narrowed the useful threshold range to {N.f2(_dca_ext[0])} to "
  f"{N.f2(_dca_ext[1])}. Any clinical use would require recalibration in the target population, "
  f"and reporting a probability without it would mislead. Second, ten stable features in a "
  f"logistic regression recovered {N.f2(N.pars('ANY disease', 10)['recovery_pct'])}% of the "
  f"chance-corrected discrimination of a {N.comma(N.N_FEATURES)}-feature model. Where the "
  f"obstacle to use is the extraction pipeline rather than the ECG, a short measurement list is "
  f"the more plausible route to the bedside.",

  "##Study limitations",
  f"The derivation cohort was assembled as a diagnostic case-control study: cases were "
  f"hospitalized children with structural disease and controls were children from general care "
  f"selected for the absence of clinical heart disease, who did not undergo echocardiography. "
  f"This spectrum, rated at high risk of bias in the PROBAST participant domain, inflates "
  f"internal discrimination relative to a consecutive series of children referred with "
  f"suspected heart disease. The case fraction ({N.pct(N.CASE_FRACTION)}%) was fixed by design, "
  f"so predictive values do not transfer to screening or referral settings.",

  f"In the derivation cohort {N.PCT_POSTSURGICAL}% of children with structural disease had "
  f"undergone prior cardiac surgery whereas controls were unoperated, so postsurgical "
  f"conduction abnormalities such as right bundle branch block after ventriculotomy could "
  f"inflate discrimination. A model derived in the external cohort, where operated children are "
  f"far less prevalent, achieved a ROC-AUC of 0.794 (95% CI, 0.719–0.862) in the 60 "
  f"non-surgical derivation cases, indicating that the learned patterns persist in unoperated "
  f"children; the small number of such cases limits the precision of that estimate.",

  f"The primary estimator was selected on nested cross-validation within the training "
  f"partition, which is the correct procedure, but the comparison between estimators was also "
  f"inspected on the held-out test set when this analysis was added, so the margin between "
  f"LightGBM and the random forest on that test set should be read as descriptive. LightGBM was "
  f"also the weakest of the three for the left atrium in nested cross-validation "
  f"({N.f3(N.nested('Left atrial'))} against {N.f3(N.nested('Left atrial', N.XGB))} for XGBoost "
  f"and {N.f3(N.nested('Left atrial', N.RF_TUNED))} for the tuned forest), although its "
  f"held-out left atrial performance was in line with the others "
  f"({N.f3(N.test_auc('Left atrial')['test_auc'])}); applying one estimator family to all five "
  f"outcomes was a pre-specified simplification, not an outcome-by-outcome optimum.",

  f"With {N.comma(N.N_FEATURES)} candidate features and {N.N_TRAIN_EVENTS} training events, the "
  f"analysis sits roughly "
  f"{int(round(float(_riley_full['required_N']) / N.N_TRAIN))} times below the participant "
  f"number the Riley criteria require, and feature-level rankings are correspondingly unstable "
  f"across bootstrap resamples. Predicted probabilities are miscalibrated by construction "
  f"because the case fraction was fixed. Derivation was at a single tertiary center, although "
  f"transportability was assessed in an independent cohort recorded on different hardware. "
  f"Automated P-wave duration agreed only modestly with manual calipers (r = 0.281; bias "
  f"+5.44 ms), reflecting delineation of low-voltage pediatric atrial signals. Transportable "
  f"features and external operating points were chosen in one half of the external cohort, so "
  f"those estimates rest on the {N.comma(int(_tr_any['n_reporting']))} children of the "
  f"reporting half. The lesion list for left atrial manifesting disease was revised during "
  f"preliminary analysis, from mitral disease, ventricular septal defect and patent ductus "
  f"arteriosus to direct loading lesions, after the initial set proved non-specific. Finally, "
  f"the external reference standard is itself a crosswalk from diagnosis codes, so the "
  f"ECG-manifesting definition improves on it without being an imaging standard.",
]

MAIN.append(("Discussion", DISCUSSION))

MAIN.append(("Conclusions", [
  f"Fully automated analysis of a single 12-lead ECG detects expert-adjudicated pediatric "
  f"structural heart disease and continues to do so in an independent cohort when the "
  f"comparison diagnosis describes a lesion that loads or injures the heart. Across three "
  f"estimator families the internal-to-external gap was unchanged, while changing the external "
  f"case definition recovered most of it, so the limit on cross-cohort accuracy in this setting "
  f"is the reference standard rather than model capacity. Internal estimates are optimistic "
  f"because of the case-control design and the predominance of operated children, and predicted "
  f"probabilities require recalibration before use. Evaluating pediatric ECG machine learning "
  f"therefore requires physiology-anchored reference standards, and prospective validation in "
  f"unoperated children is the next step."]))

# ------------------------------------------------------------------ Statements and Declarations
DECLARATIONS = [
 ("Funding", "This research received no specific grant from any funding agency in the public, "
             "commercial or not-for-profit sectors."),
 ("Competing interests", "The authors have no competing interests to declare that are relevant "
                         "to the content of this article."),
 ("Ethics approval", "The study was performed in accordance with the ethical standards of the "
                     "1964 Declaration of Helsinki and its later amendments. The derivation "
                     "cohort was approved by the institutional review board of Instituto Dante "
                     "Pazzanese de Cardiologia, S\u00e3o Paulo, Brazil (protocol "
                     "75263023.5.0000.5462), under which the parent CHILDHEART studies were "
                     "conducted and of which the present analysis forms part. The external "
                     "cohort is a published, de-identified research database whose collection "
                     "was reviewed and approved by the Ethics Committee for Scientific Research "
                     "and Clinical Trials of the First Affiliated Hospital of Zhengzhou "
                     "University (2024-KY-0221-003 and 2025-KY-0369-001).[[zzu]]"),
 ("Consent to participate", "The institutional review board of Instituto Dante Pazzanese de "
                            "Cardiologia waived the requirement for informed consent for this "
                            "retrospective analysis of de-identified records, including the "
                            "tele-electrocardiography comparison group. For the external cohort, "
                            "the approving committee likewise agreed to waive the signing of "
                            "informed consent and required the electrocardiographic data to be "
                            "de-identified before public release.[[zzu]] No identifiable "
                            "participant information left either institution."),
 ("Consent to publish", "The manuscript contains no individually identifiable participant data "
                        "or images."),
 ("Data availability", "The CHILDHEART derivation data (12-lead recordings, extracted features "
                       "and chamber-level adjudications) contain potentially identifying "
                       "pediatric clinical information and are not publicly available. "
                       "De-identified feature tables and outcome labels are available from the "
                       "corresponding author on reasonable request, subject to approval by the "
                       "Instituto Dante Pazzanese research ethics committee and a data-use "
                       "agreement. The external cohort is the published Zhengzhou University "
                       "pediatric 12-lead ECG database, deposited by its authors on figshare in "
                       "WaveForm DataBase format and used here under its stated terms.[[zzu]]"),
 ("Code availability", f"Code for model derivation, internal evaluation, external validation and "
                       f"every analysis reported here is available at "
                       f"https://github.com/{GH_USER}/{GH_REPO}. Feature extraction uses the "
                       f"ecg-vcg-viewer engine at delineator D2, extractor E3 and measurement "
                       f"contract M2 (commit {N.ENGINE_COMMIT}); the engine was developed by our "
                       f"group and is available from the authors on reasonable request. Because "
                       f"measured values move between extractor versions, the commit is required "
                       f"to reproduce the numbers reported here."),
 ("Author contributions", "Conceptualization, methodology, resources, software and writing of "
                          "the original draft: J.N.A.N. Writing, review and editing: all "
                          "authors. All authors read and approved the final manuscript and agree "
                          "to be accountable for all aspects of the work."),
 ("Use of large language models", "A large language model was used to "
                                  "assist with analysis code, table and figure generation, "
                                  "language editing, and a reporting-standard audit that "
                                  "prompted several of the sensitivity analyses reported in "
                                  "Material and methods. All study design, analytical "
                                  "decisions, interpretation and scientific content were "
                                  "determined by the authors, who verified all code and results "
                                  "and take full responsibility for their accuracy. No "
                                  "identifiable participant information was shared with the "
                                  "tool."),
]

BACK = DECLARATIONS
