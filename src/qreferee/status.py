"""Shared verdict/status vocabulary used across the whole package.

A single source of truth so the CHSH certifier and the Study verdict speak the same
words (no ``VIOLATION_CERTIFIED`` vs ``CERTIFIED`` drift) and the README taxonomy
matches the code exactly.
"""

from __future__ import annotations

CERTIFIED = "CERTIFIED"
NOT_CERTIFIED = "NOT_CERTIFIED"
UNDERPOWERED = "UNDERPOWERED"
ASSUMPTIONS_UNMET = "ASSUMPTIONS_UNMET"

__all__ = ["CERTIFIED", "NOT_CERTIFIED", "UNDERPOWERED", "ASSUMPTIONS_UNMET"]
