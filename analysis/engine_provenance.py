"""Where the numbers came from, READ rather than typed.

An engine string typed into a report header is a constant like any other, and constants
here have twice outlived the extraction they described: `ec3e2f6` stayed in five report
headers of the companion score paper after its cohort had been re-extracted, and this
study's manuscript figures carried a June LV AUC of 0.944 through two re-extractions while
the tables said 0.936.

So the provenance is derived at run time from the four things that actually determine a
measurement: the delineator, the feature extractor, the measurement contract, and the
commit of the engine that produced the feature tables.

That last one is deliberately the TABLE's commit (batch_source.extraction_commit), not the
engine's current HEAD. During the 2026-09-09 migration the viewer checkout moved onto a
feature branch mid-run, and a report that prints HEAD would have stamped the results with a
commit that had nothing to do with them. When the checkout differs from the table's commit
the string says so, rather than quietly asserting either one.
"""
import os, re, subprocess

VIEWER = os.environ.get('ECG_VIEWER', os.path.expanduser('~/ecg-vcg-viewer'))


def _const(path, name):
    try:
        m = re.search(rf"{name}\s*=\s*['\"]([^'\"]+)['\"]", open(os.path.join(VIEWER, path)).read())
        return m.group(1) if m else '?'
    except OSError:
        return '?'


def _git(*args):
    try:
        return subprocess.run(['git', '-C', VIEWER, *args],
                              capture_output=True, text=True, timeout=10).stdout.strip() or '?'
    except Exception:
        return '?'


def commit():
    return _git('rev-parse', '--short', 'HEAD')


def branch():
    return _git('rev-parse', '--abbrev-ref', 'HEAD')


def engine_string(vint=None):
    """e.g. "delineator D2, extractor E3, contract M2; feature table `v4_E3`, extracted at `150be8b`"."""
    if vint is None:
        try:
            from batch_source import vintage
            vint = vintage()
        except Exception:
            vint = os.environ.get('CH_VINTAGE', 'v4_E3')
    try:
        from batch_source import extraction_commit
        made_by = extraction_commit(vint)
    except Exception:
        made_by = '?'
    s = (f"delineator {_const('js/classicalDelineate.js', 'DELINEATOR_VERSION')}, "
         f"extractor {_const('js/extractorVersion.js', 'EXTRACTOR_VERSION')}, "
         f"measurement contract {_const('js/canonicalMeasure.js', 'MEASUREMENT_VERSION')}; "
         f"feature table `{vint}`, extracted at `{made_by}`")
    here = commit()
    if here != made_by:
        s += f" (viewer now checked out at `{here}` on `{branch()}`)"
    return s


if __name__ == '__main__':
    print(engine_string())
