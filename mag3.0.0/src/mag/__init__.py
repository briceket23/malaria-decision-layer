"""mag — patient-level aggregation rules for automated malaria microscopy.

A rewrite. The predecessor grew by accretion around a working analysis and
ended up with two properties that cost an entire review cycle: it could
generate a cohort that looked like data, and its results file could exist
without being able to say where it came from. Neither is possible here.

The shape of the code follows the shape of the argument. A rule is the triple
the manuscript defines --- an evidence functional, a screening statistic and a
threshold --- and `rules.py` holds them as data rather than as branches, so the
paper and the registry can be read side by side. Everything downstream of
`detect` is a pure function of a cached detection table, so the whole
experiment runs on a laptop once the GPU pass is done.

Three invariants hold everywhere and are tested rather than trusted:

  1. No module in this package can invent an observation. There is no
     generator, no default cache, no fallback that supplies a value when a
     measurement is missing. A missing input is an error with a remedy.
  2. Every result carries its provenance, and `report` refuses to write for a
     provenance that cannot name the checkpoint hashes and inference shards it
     descends from.
  3. Nothing that decides an outcome is written in prose. Directional claims
     leave the analysis as generated sentences, so a document cannot assert a
     direction its run did not produce.
"""

from __future__ import annotations

#: Kept in step with ``pyproject.toml`` by ``test_version_matches_pyproject``.
#: A literal rather than an ``importlib.metadata`` lookup, because the version
#: is stamped into every ``provenance.json`` and must be readable from a source
#: tree that was never installed.
__version__ = "3.0.0"

__all__ = ["__version__", "build_id"]


def build_id(root=None) -> str:
    """A short fingerprint of this package's own source.

    Printed on the first line of every run and written into every result, so
    that console output pasted into an email identifies the code that produced
    it. Line endings are normalised, so a release has one fingerprint on every
    platform.
    """
    import hashlib
    from pathlib import Path

    here = Path(root) if root is not None else Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(here.glob("*.py")):
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:10]
