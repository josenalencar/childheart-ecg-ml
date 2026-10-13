# Pediatric Cardiology submission — how it is built

Upload order (Original Article, submitted as "Research"): 01 manuscript (title page, abstract,
keywords, text, Statements and Declarations, references, tables, figure legends) · 02–03 figures
(vector PDF; 600 dpi TIFF as backup) · 04 Online Resource 1 (supplement PDF) · 05–06 reporting
checklists · 07 cover letter.


## Rebuild
```bash
cp ~/Zotero/zotero.sqlite /private/tmp/zsrc_pedcard.sqlite      # Zotero locks the original
python3 _source/build_all.py "<this folder>" /private/tmp/zsrc_pedcard.sqlite
"/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge" --headless --disable-gpu \
  --no-pdf-header-footer --print-to-pdf="04 Online Resource 1 - ESM_1.pdf" \
  "file://$PWD/_source/supplement.html"
python3 _source/make_fig1.py && cp _source/Fig1.pdf "02 Fig1.pdf" && cp _source/Fig1.tiff "02 Fig1.tif"
python3 _source/make_fig2_panelb.py && python3 _source/make_fig2.py \
  && cp _source/Fig2.pdf "03 Fig2.pdf" && cp _source/Fig2.tiff "03 Fig2.tif"
```
Rebuilding overwrites `01 Manuscript.docx`: open it in Word and click **Zotero > Refresh**
afterwards, then **Unlink Citations** on the copy you upload (the journal asks authors not to
use field functions; keep the linked version in `_source/`).

## Where the words and the numbers live
| file | what |
|---|---|
| `nums.py` | **Every number in the paper**, read from `results/tripod_analyses/`. Nothing is typed. `one()` raises if a row is not unique, so a renamed estimator fails the build instead of printing a wrong value. Run `python3 nums.py` for a summary table. |
| `content.py` | Manuscript text, LightGBM-primary. Citations as `[[key]]`. |
| `supp.py` | Tables 1–3, figure legends, eMethods 1–4, eTables 1–12. |
| `checklists.py` | Where each TRIPOD+AI (52) and STARD (34) item is reported, plus the cover letter. |
| `zot.py` | The 28 Zotero item keys, 25 recovered from the previous manuscript's live fields. |
| `build_all.py` | Assembles everything; prints the citation order and the abstract word count. |

## Primary model
LightGBM, chosen on training data alone by the highest nested cross-validated ROC-AUC for the
primary outcome (0.983 vs 0.979 XGBoost and 0.976 random forest,
`results/tripod_analyses/ceiling/ceiling_nested_cv.csv`). The random forest and XGBoost are
comparators throughout. The one place LightGBM is weakest — left atrial nested CV 0.869 against
0.912 for the others — is disclosed in the limitations.

## Style
Citations are bracketed numerals in citation order, CSL **Springer Basic (numeric, brackets)**,
installed at `~/Zotero/styles/springer-basic-brackets.csl` on 2026-10-13. Times New Roman 12 pt,
double spaced, continuous line numbers, page numbers in the footer.

## Figures
Both figures are built with the group's in-house `scifig` composite toolkit (set `SCIFIG_PATH`),
so they share one visual system: blue header bands, square flow boxes, italic *n = …*, Arial only.
- **Fig 1** `make_fig1.py` — participant flow for both cohorts (STARD item 19).
- **Fig 2** `make_fig2.py` — panel a, what each external case definition counts as a case;
  panel b, `make_fig2_panelb.py` embedded as vector at scale 1.0.
Both are 174 mm wide (the journal's single-column-text width), vector PDF plus 600 dpi TIFF,
with only Arial faces embedded. The toolkit's lint warns that composite figures are 180 mm; 174 mm is deliberate here.
