"""Where each TRIPOD+AI and STARD 2015 item is reported in this manuscript, and the cover letter.

Item wording itself is read verbatim from items.json (both checklists are open access); this
file only says where to look. An item deliberately not met is still given a location, pointing
at the sentence that says so -- a checklist whose purpose is to let a reader find the weakness
is more useful than one with a tick in every row.
"""

M, R, D, S = "Material and methods", "Results", "Discussion", "Statements and Declarations"

TRIPOD_LOC = {
    '1':   'Title page',
    '2':   'Abstract',
    '3a':  'Introduction, paragraphs 1-3',
    '3b':  'Introduction, paragraphs 1-2 and final paragraph',
    '3c':  'NOT REPORTED. Ethnicity and socioeconomic status were not recorded in either cohort; '
           f'see {R}, Subgroups and {D}, Study limitations',
    '4':   'Introduction, final paragraph (development and external evaluation; hypothesis stated)',
    '5a':  f'{M}, Cohorts and reference standards (separate derivation and external sources)',
    '5b':  f'{M}, Cohorts and reference standards (derivation during 2023); external dates in '
           f'the source database publication',
    '6a':  f'{M}, Cohorts and reference standards (tertiary ward and intensive care; '
           f'tele-electrocardiography serving general care)',
    '6b':  f'{M}, Cohorts and reference standards; Fig. 1',
    '6c':  f'{M}, Cohorts and reference standards (82.8% of cases post-surgical); '
           f'{R}, Never, past and current chamber involvement; {D}, Study limitations',
    '7':   f'{M}, Electrocardiographic feature extraction; Online Resource 1, eMethods 1 '
           f'(identical pipeline and age-normalisation matrix applied to both cohorts)',
    '8a':  f'{M}, Cohorts and reference standards (chamber involvement at the index admission); '
           f'{M}, External case definitions',
    '8b':  f'{M}, Cohorts and reference standards (three pediatric cardiologists)',
    '8c':  f'{M}, Cohorts and reference standards (adjudicators blinded to the ECG)',
    '9a':  f'{M}, Electrocardiographic feature extraction (all extractor outputs offered); eMethods 1',
    '9b':  f'{M}, Electrocardiographic feature extraction; eMethods 1',
    '9c':  f'{M}, Electrocardiographic feature extraction (fully automated; no subjective reading)',
    '10':  f'{R}, Sample size, ranking stability and a parsimony floor; eTable 10. '
           f'The sample size was not pre-computed; see {D}, Study limitations',
    '11':  f'{M}, Electrocardiographic feature extraction (implausible values set to missing, '
           f'imputed with training-set medians); eMethods 1',
    '12a': f'{M}, Cohorts and reference standards (fixed 80/20 split); {M}, Model development',
    '12b': f'{M}, Electrocardiographic feature extraction (age-specific z scores); eMethods 1',
    '12c': f'{M}, Model development and selection of the primary estimator; eMethods 2; eTable 1',
    '12d': f'{R}, Variability of the held-out estimate and subgroups; eTable 5; eTable 6',
    '12e': f'{M}, Statistical analysis; eMethods 4',
    '12f': 'No model updating or recalibration was performed; the need for it is stated in '
           f'{R}, Calibration and clinical usefulness and in {D}',
    '12g': f'{M}, Model development (model refitted on the training partition and applied frozen); '
           f'{S}, Code availability',
    '13':  f'{M}, Model development (balanced class weights; no resampling used)',
    '14':  f'{R}, Subgroups; eTable 5. Ethnicity was not recorded, so fairness across ethnic '
           f'groups could not be assessed ({D}, Study limitations)',
    '15':  f'{M}, Model development (predicted probabilities, with two pre-specified thresholds); '
           f'Table 2',
    '16':  f'Table 1; {R}, Cohorts; {M}, External case definitions',
    '17':  f'{S}, Ethics approval',
    '18a': f'{S}, Funding',
    '18b': f'{S}, Competing interests',
    '18c': f'{M}, Study design and reporting (no protocol was published)',
    '18d': f'{M}, Study design and reporting (not registered)',
    '18e': f'{S}, Data availability',
    '18f': f'{S}, Code availability',
    '19':  f'{S}, Ethics approval (patients and the public were not involved)',
    '20a': 'Fig. 1',
    '20b': f'Table 1; {R}, Cohorts',
    '20c': f'Table 1 (derivation versus external age, sex and reference standard); '
           f'{R}, Cohorts',
    '21':  f'Fig. 1; Table 2 (events/n per outcome); {R}, Internal discrimination',
    '22':  f'{S}, Code availability (trained model and feature list available with the code); '
           f'eTable 1 for hyperparameters',
    '23a': f'Table 2; Table 3; {R}, Internal discrimination; {R}, Subgroups; eTable 5',
    '23b': 'Not examined; the derivation data come from a single centre '
           f'({D}, Study limitations)',
    '24':  'No model updating was performed',
    '25':  f'{D}, paragraphs 1-6',
    '26':  f'{D}, Study limitations',
    '27a': f'{M}, Electrocardiographic feature extraction (implausible values); '
           f'{D} (recalibration required before use)',
    '27b': f'{D} (fully automated pipeline; ten-feature parsimony floor as the practical route)',
    '27c': 'Conclusions (prospective validation in unoperated children)',
}

STARD_LOC = {
    '1':   'Title page',
    '2':   'Abstract',
    '3':   'Introduction, paragraphs 1-3',
    '4':   'Introduction, final paragraph (hypothesis)',
    '5':   f'{M}, Cohorts and reference standards (retrospective analysis of existing cohort data)',
    '6':   f'{M}, Cohorts and reference standards; Fig. 1',
    '7':   f'{M}, Cohorts and reference standards (consecutive admissions; separately recruited '
           f'control series)',
    '8':   f'{M}, Cohorts and reference standards; Fig. 1',
    '9':   f'{M}, Cohorts and reference standards (single tertiary centre; tele-electrocardiography '
           f'services; external database)',
    '10a': f'{M}, Electrocardiographic feature extraction; {M}, Model development; eMethods 1-2',
    '10b': f'{M}, Cohorts and reference standards (adjudication by three pediatric cardiologists); '
           f'{M}, External case definitions (ICD-10 crosswalk)',
    '11':  f'{M}, Cohorts and reference standards (echocardiography the principal but not the sole '
           f'source); {D}, Study limitations',
    '12a': f'{M}, Model development (Youden and >=95% specificity thresholds, both fixed on '
           f'out-of-fold training predictions)',
    '12b': f'{M}, Cohorts and reference standards; {M}, External case definitions',
    '13a': f'{M}, Cohorts and reference standards (adjudicators blinded to the ECG)',
    '13b': f'{M}, Electrocardiographic feature extraction (fully automated index test; no reader)',
    '14':  f'{M}, Statistical analysis',
    '15':  f'{M}, Electrocardiographic feature extraction (implausible values set to missing)',
    '16':  'Fig. 1 (signal-extraction failures and ineligible tracings)',
    '17':  f'{R}, External case definition; {R}, Subgroups; eTable 5',
    '18':  f'{R}, Sample size, ranking stability and a parsimony floor; eTable 10. '
           f'Not pre-computed; see {D}, Study limitations',
    '19':  'Fig. 1',
    '20':  f'Table 1; {R}, Cohorts',
    '21a': f'{R}, External case definition; Fig. 2a; Table 3',
    '21b': f'{R}, Never, past and current chamber involvement; eTable 11',
    '22':  f'{M}, Cohorts and reference standards (one ECG per child at the index admission); '
           f'{D}, Study limitations',
    '23':  f'Table 2 (counts at each pre-specified threshold)',
    '24':  f'Table 2; Table 3; eTable 3 (DeLong comparisons with confidence intervals)',
    '25':  f'{R}, Cohorts; none reported in either cohort',
    '26':  f'{D}, Study limitations',
    '27':  f'{D}, final paragraphs; Conclusions',
    '28':  f'{M}, Study design and reporting (not registered)',
    '29':  f'{M}, Study design and reporting (no published protocol)',
    '30':  f'{S}, Funding',
}

COVER = [
    "To the Editor-in-Chief, Pediatric Cardiology",
    "",
    "Dear Editor,",
    "",
    "We submit for your consideration an Original Article entitled “Machine Learning Detects "
    "Pediatric Structural Heart Disease From the 12-Lead Electrocardiogram: External Validation "
    "in 10,352 Children”.",
    "",
    "Echocardiography is often unavailable where a child is first seen, while the 12-lead ECG is "
    "almost everywhere, so an automated ECG classifier is an attractive triage tool. The "
    "pediatric literature on this question is dominated by single-centre models, and the larger "
    "studies judge their models against administrative diagnosis codes. Our study was designed "
    "to ask what that choice costs.",
    "",
    "We derived models in 1,637 Brazilian children whose chamber involvement was adjudicated by "
    "three pediatric cardiologists, and applied them unchanged to 10,352 Chinese children "
    "labelled by an ICD-10 crosswalk. Rather than report a single external figure, we varied two "
    "things independently: the learning algorithm and the external case definition. Three "
    "estimator families, tuned identically, separated internally but produced an "
    "internal-to-external difference that was essentially identical (0.285, 0.290 and 0.284). "
    "With the models untouched, restricting cases to lesions that load or injure the heart "
    "recovered external discrimination from 0.68 to 0.91. The ceiling on cross-cohort accuracy "
    "here is set by what the comparison diagnosis means, not by model capacity, and the internal "
    "advantage of the stronger algorithm largely disappeared once the label described physiology.",
    "",
    "We think the practical message suits your readership: a pediatric ECG artificial "
    "intelligence study reporting external accuracy against diagnosis codes alone cannot "
    "separate a model that has failed from a model being asked the wrong question, and should "
    "report at least one physiology-anchored case definition alongside.",
    "",
    "We have tried to be plain about the limits. The derivation cohort was assembled as a "
    "case-control study and its case fraction is fixed by design, so internal estimates are "
    "optimistic; 82.8% of cases had undergone cardiac surgery; the external labels are "
    "diagnosis codes rather than imaging; predicted probabilities are miscalibrated externally "
    "and would need recalibration before use; and with 1,047 candidate features the feature "
    "rankings are unstable, which we report rather than smooth over. The study follows TRIPOD+AI "
    "and STARD 2015 and both checklists are attached.",
    "",
    "An abstract reporting part of this work has been accepted for presentation at the American "
    "College of Cardiology Annual Scientific Session 2027; no part of the manuscript has been "
    "published, and it is not under consideration elsewhere. All authors have approved the "
    "submission. We are submitting through the subscription route and are not requesting open "
    "access publication.",
    "",
    "Thank you for considering our work.",
    "",
    "Yours sincerely,",
    "",
    "José Nunes de Alencar Neto, MD, on behalf of all authors",
    "Instituto Dante Pazzanese de Cardiologia, São Paulo, Brazil",
    "josenunesalencar@gmail.com",
]
