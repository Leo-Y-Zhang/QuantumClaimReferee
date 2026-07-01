"""Framework-agnostic ingestion of raw measurement counts.

A ``CountsDataset`` is one schema every SDK's results reduce to: a set of named
measurement settings, each a ``{bitstring: shots}`` histogram. Qiskit, Cirq and
PennyLane all expose counts as such dicts, so adapters are thin and qreferee keeps
its dependency footprint to numpy/scipy.

Default-deny is baked in: the assumed measurement (POVM) must be declared, because a
silent "assume ideal projective measurement" would bias every downstream estimate no
matter how correct the statistics on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["CountsDataset"]


@dataclass(frozen=True)
class CountsDataset:
    settings: dict[str, dict[str, int]]
    povm: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.settings:
            raise ValueError("at least one measurement setting is required")
        for label, hist in self.settings.items():
            if not hist:
                raise ValueError(f"setting {label!r} has no counts")
            width = len(next(iter(hist)))
            for bitstring, c in hist.items():
                if c < 0:
                    raise ValueError(f"negative count in setting {label!r}")
                if len(bitstring) != width or any(ch not in "01" for ch in bitstring):
                    raise ValueError(
                        f"setting {label!r}: bitstrings must be equal-length 0/1 strings"
                    )

    def require_povm(self) -> str:
        """Return the declared POVM or raise -- the default-deny gate."""
        if not self.povm:
            raise ValueError(
                "POVM undeclared: refusing to estimate. Pass povm=... "
                "(e.g. 'ideal_projective') to affirm the measurement model."
            )
        return self.povm

    def shots(self, setting: str) -> int:
        return sum(self.settings[setting].values())

    def counts(self, setting: str) -> dict[str, int]:
        return dict(self.settings[setting])

    def probability(self, setting: str, bitstring: str) -> float:
        n = self.shots(setting)
        return self.settings[setting].get(bitstring, 0) / n if n else 0.0

    @classmethod
    def from_counts(
        cls,
        counts: dict[str, int],
        *,
        setting: str = "default",
        povm: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> "CountsDataset":
        """Build from a single ``{bitstring: shots}`` histogram."""
        return cls({setting: dict(counts)}, povm=povm, meta=meta or {})

    @classmethod
    def from_qiskit(
        cls,
        result: Any,
        *,
        setting: str = "default",
        povm: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> "CountsDataset":
        """Build from a Qiskit ``Result`` (duck-typed ``.get_counts()``)."""
        if not hasattr(result, "get_counts"):
            raise TypeError("expected a Qiskit-like object exposing .get_counts()")
        raw = result.get_counts()
        if isinstance(raw, list):
            raw = raw[0]
        # normalise bitstrings (strip spaces used by multi-register results)
        hist = {str(k).replace(" ", ""): int(v) for k, v in raw.items()}
        return cls({setting: hist}, povm=povm, meta=meta or {})
