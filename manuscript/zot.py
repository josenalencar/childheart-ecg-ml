"""Zotero item keys for this manuscript, and CSL extraction from a copy of zotero.sqlite.

The 25 keys below were recovered from the live citation fields of the previous manuscript
(Manuscript/CHILDHEART_RF_Manuscript_v5_2026-09-25.docx) so the reference list carries over
intact; the three reporting-standard items (TRIPOD+AI, STARD, the ZZU dataset) come from the
sibling LV-score package. Reading the keys from the old document rather than retyping them is
what keeps the bibliography identical across the two papers.

Zotero must be running for the live-API path (http://127.0.0.1:23119); build_all.py reads a
COPY of zotero.sqlite instead, because Zotero holds a lock on the original.
"""
import sqlite3, re

KEYS = {
    # burden, normal standards, and manual interpretation
    'liu2019':      'CI53KVQ2',   # Liu 2019, global birth prevalence of congenital heart defects
    'davignon':     'GEPGDRVS',   # Davignon 1980, normal ECG standards for infants and children
    'evans':        'WWRTGVCD',   # Evans 2010, simplified pediatric ECG interpretation
    'milestones':   'JKTAAAS8',   # Alencar 2025, age-related ECG milestones (the parent CHILDHEART paper)
    # machine learning on the ECG
    'attia':        'LFF3FE24',   # Attia 2021, external validation of a deep learning ECG algorithm
    'alzaiti':      'CEP7WQW2',   # Al-Zaiti 2023, machine learning for ECG diagnosis and risk stratification
    'mori':         'IHDGHPUU',   # Mori 2021, atrial septal defect from the ECG with deep learning (Pediatr Cardiol)
    'lee2025':      'Q8EG36SL',   # Lee 2025, prediction of significant congenital heart disease
    'pachiyannan':  'A4UYGU5S',   # Pachiyannan 2024, ML-based early detection
    'devries':      'UDYVPE8E',   # de Vries 2023, fetal electrocardiography and artificial intelligence
    # administrative labels
    'khan':         'INLKWDQT',   # Khan 2018, limited accuracy of administrative data
    # reporting standards and risk of bias
    'tripod':       'CT9FI87B',   # Collins 2015, TRIPOD
    'tripodai':     'KRQKF5VF',   # Collins 2024, TRIPOD+AI
    'probast':      '8XBBYZ3C',   # Wolff 2019, PROBAST
    'stard':        'D9WEKHCM',   # Bossuyt 2015, STARD 2015
    # the ECG features the models are built from
    'bergfeldt':    'H6DLV9QT',   # Bergfeldt 2020, spatial peak and mean QRS-T angles
    'nussinovitch': '37JE62XH',   # Nussinovitch 2020, total cosine R-to-T
    'casale':       'KU4YS7BU',   # Casale 1985, Cornell criteria
    'peguero':      'A62EVG33',   # Peguero 2017, Peguero-Lo Presti criteria
    'romhilt':      'ZHYGRHLS',   # Romhilt 1968, point-score system
    'prineas':      'M364UP2P',   # Prineas 2010, Minnesota Code manual
    # methods
    'xgboost':      'P8UWG7KD',   # Chen & Guestrin 2016, XGBoost
    'lightgbm':     '6F6HL99C',   # Ke 2017, LightGBM
    'delong':       'S8JKF9NX',   # DeLong 1988, comparing correlated ROC curves
    'riley':        '6VKSKNVV',   # Riley 2020, sample size for a clinical prediction model
    'treeshap':     'W26VFWXD',   # Lundberg 2020, TreeSHAP
    'vickers':      '7R79XMXD',   # Vickers & Elkin 2006, decision curve analysis
    # the external cohort
    'zzu':          'PYXGW3IC',   # the Zhengzhou University paediatric ECG database
}

TYPE = {'journalArticle': 'article-journal', 'dataset': 'dataset', 'book': 'book',
        'bookSection': 'chapter', 'webpage': 'webpage', 'conferencePaper': 'paper-conference',
        'preprint': 'article', 'report': 'report'}


def csl(db, key):
    """The CSL-JSON for one library item, straight out of zotero.sqlite."""
    row = db.execute("select i.itemID, t.typeName from items i join itemTypes t "
                     "on t.itemTypeID=i.itemTypeID where i.key=? and i.libraryID=1",
                     (key,)).fetchone()
    if row is None:
        raise KeyError(f'zot.py: item {key} is not in this Zotero library copy')
    iid, typ = row
    F = dict(db.execute(
        "select f.fieldName, v.value from itemData d join fields f on f.fieldID=d.fieldID "
        "join itemDataValues v on v.valueID=d.valueID where d.itemID=?", (iid,)).fetchall())
    au = db.execute(
        "select c.firstName, c.lastName, ct.creatorType from itemCreators ic "
        "join creators c on c.creatorID=ic.creatorID join creatorTypes ct "
        "on ct.creatorTypeID=ic.creatorTypeID where ic.itemID=? order by ic.orderIndex",
        (iid,)).fetchall()
    d = {'id': iid, 'type': TYPE.get(typ, 'article-journal'), 'title': F.get('title', '')}
    m = {'publicationTitle': 'container-title', 'journalAbbreviation': 'container-title-short',
         'volume': 'volume', 'issue': 'issue', 'pages': 'page', 'DOI': 'DOI', 'ISSN': 'ISSN',
         'url': 'URL', 'publisher': 'publisher', 'repository': 'publisher',
         'proceedingsTitle': 'container-title'}
    for z, c in m.items():
        if F.get(z): d[c] = F[z]
    if F.get('date'):
        ds = re.findall(r'\d+', F['date'].split(' ')[0])
        if ds: d['issued'] = {'date-parts': [[int(x) for x in ds[:3]]]}
    d['author'] = [{'family': l, 'given': f} for f, l, t in au if t == 'author']
    return iid, F, d


if __name__ == '__main__':
    import sys, shutil, os, tempfile
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/Zotero/zotero.sqlite')
    tmp = os.path.join(tempfile.gettempdir(), 'zcopy_pedcard.sqlite')
    shutil.copy(src, tmp)
    db = sqlite3.connect(tmp)
    bad = []
    for k, v in KEYS.items():
        try:
            iid, F, d = csl(db, v)
            au = d['author'][0]['family'] if d['author'] else '?'
            yr = d.get('issued', {}).get('date-parts', [['?']])[0][0]
            print(f"{k:14s} {v}  {au:16s} {yr}  {(d.get('container-title-short') or d.get('container-title') or '')[:34]:34s} "
                  f"{'DOI ok' if d.get('DOI') else 'NO DOI'}")
        except KeyError as e:
            bad.append(k); print(f"{k:14s} {v}  MISSING -- {e}")
    print(f"\n{len(KEYS) - len(bad)}/{len(KEYS)} items resolved")
    if bad: print('missing:', ', '.join(bad))
