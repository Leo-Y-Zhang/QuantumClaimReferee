"""Deterministic referee report -- the artifact you attach to a paper or review.

The report is a headline verdict over the full per-hypothesis numbers, plus a
reproducibility block (tool + dependency versions and a content hash of the inputs)
and an explicit caveats section. It contains no timestamps or randomness, so the same
inputs always render byte-identical -- a property a referee can rely on.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import scipy

from ._version import __version__
from .verdict import Verdict

__all__ = ["referee_report"]

_DEFAULT_CAVEATS = (
    "Estimates assume the declared measurement model (POVM); no SPAM correction.",
    "CHSH validity assumes settings randomised per round and no-signaling.",
    "Intervals cover linear functionals only; no full-state region is claimed.",
)


def _content_hash(verdict: Verdict) -> str:
    payload = [
        {"name": r.name, "raw_p": r.raw_p, "status": r.status} for r in verdict.results
    ]
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def referee_report(
    verdict: Verdict,
    *,
    title: str = "minos report",
    caveats: tuple[str, ...] = _DEFAULT_CAVEATS,
    meta: dict | None = None,
) -> str:
    """Render a :class:`~minos.verdict.Verdict` as a deterministic referee report.

    The output is a headline verdict over the per-hypothesis numbers, a reproducibility
    block (tool + dependency versions and a content hash of the inputs) and a caveats
    section. It contains no timestamps or randomness, so identical inputs render
    byte-identical text. Pass ``meta`` for extra header lines (rendered sorted).
    """
    bar = "=" * 68
    lines = [bar, f" {title}", bar, ""]
    lines.append(f" VERDICT: {verdict.classification}")
    lines.append(
        f" policy: alpha={verdict.alpha} | correction={verdict.correction} | default-deny"
    )
    if meta:
        for key in sorted(meta):
            lines.append(f" {key}: {meta[key]}")
    lines.append("")
    lines.append(f" {'HYPOTHESIS':<34}{'raw p':>12}{'adj p':>12}  STATUS")
    lines.append(f" {'-' * 34}{'-' * 12}{'-' * 12}  {'-' * 16}")
    for r in verdict.results:
        raw = "n/a" if r.raw_p != r.raw_p else f"{r.raw_p:.2e}"
        adj = "n/a" if r.adjusted_p != r.adjusted_p else f"{r.adjusted_p:.2e}"
        lines.append(f" {r.name:<34}{raw:>12}{adj:>12}  {r.status}")
        if r.estimate:
            lines.append(f"   {r.estimate}")
    lines.append("")
    lines.append(" REPRODUCIBILITY")
    lines.append(
        f"   minos {__version__} | numpy {np.__version__} | scipy {scipy.__version__}"
    )
    lines.append(f"   inputs sha256:{_content_hash(verdict)}")
    lines.append("")
    lines.append(" CAVEATS (read before citing)")
    for c in caveats:
        lines.append(f"   - {c}")
    lines.append(bar)
    return "\n".join(lines)
