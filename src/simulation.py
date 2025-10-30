"""Simulation glue connecting data and rules.

This module exposes a single function `run_simulation` which accepts a mapping
of sector -> CO₂ values and a mapping of sector -> user-provided percentage
changes, normalizes the inputs, calls the rules engine, and returns an updated
mapping of sector -> new CO₂ values.
"""

from typing import Dict, Any

from rules_engine import apply_rules


def _normalize_changes(changes: Dict[str, Any]) -> Dict[str, float]:
    """Normalize user-provided sector changes.

    Accepts either decimal fractions (e.g. -0.2 for -20%) or percent values
    (e.g. -20). If abs(value) > 1 we treat it as percent and divide by 100.
    Returns mapping sector -> decimal fraction.
    """
    normalized: Dict[str, float] = {}
    for s, v in (changes or {}).items():
        try:
            num = float(v)
        except Exception:
            num = 0.0
        # If user passed percentages like -20 or 20, convert to fraction
        if abs(num) > 1:
            num = num / 100.0
        normalized[s] = num
    return normalized


def run_simulation(sector_data: Dict[str, float], sector_changes: Dict[str, Any],
                   rules_map: Dict[str, Dict[str, float]] = None) -> Dict[str, float]:
    """
    Run the simulation by applying direct sector_changes to sector_data and
    then propagating effects using the rules engine.

    - sector_data: dict sector -> CO₂ absolute value
    - sector_changes: dict sector -> percent change (either decimal or percent)
    - rules_map: optional rules dictionary to override the default in rules_engine

    Returns a new dict sector -> updated CO₂ absolute value.
    """
    if not isinstance(sector_data, dict):
        raise ValueError('sector_data must be a dict mapping sector->value')

    normalized_changes = _normalize_changes(sector_changes or {})

    # Call the rules engine which returns updated absolute values
    updated = apply_rules(sector_data, normalized_changes, rules_map=rules_map)
    return updated
