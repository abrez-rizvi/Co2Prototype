"""Rules engine for sector interactions.

This module provides a simple rules-based propagation engine. The rules map
describes how a change in one sector cascades to others. Coefficients may be
positive or negative and represent how an absolute change in sector A
influences sector B. The engine applies direct user changes (percentage
changes) and then propagates cascading effects iteratively until convergence
or a maximum number of iterations is reached.

Example rules dict:
rules = {
  "transport": {"industry": -0.3, "energy": -0.1},
  "energy": {"industry": -0.2, "infrastructure": -0.1},
  "industry": {"energy": 0.2},
  "infrastructure": {"energy": 0.1}
}

Function:
- apply_rules(sector_data, sector_changes)

Inputs:
- sector_data: dict[str, float] — current CO₂ values per sector
- sector_changes: dict[str, float] — direct percentage changes to apply (e.g. -0.2 = -20%)

Returns updated dict of CO₂ values after applying rules.
"""

from typing import Dict, Any

# Default rule set (can be overridden by callers)
# Coefficients represent fraction of change propagated: negative = reduction cascade, positive = increase
# Keep values small (< 0.15) to avoid unrealistic amplification
rules: Dict[str, Dict[str, float]] = {
    "transport": {"industry": -0.05, "power": -0.02},
    "industry": {"power": -0.08},
    "residential": {"power": -0.03},
    "power": {},  # power doesn't cascade to others
}


def apply_rules(sector_data: Dict[str, float], sector_changes: Dict[str, float],
                rules_map: Dict[str, Dict[str, float]] = None,
                max_iterations: int = 10,
                tol: float = 1e-6) -> Dict[str, float]:
    """
    Apply direct sector changes and propagate cascading effects using rules_map.

    - sector_data: mapping sector -> current CO₂ value (absolute)
    - sector_changes: mapping sector -> percentage change as decimal (e.g. -0.2)
    - rules_map: optional rules dictionary; if None, the module-level `rules` is used
    - max_iterations: max propagation rounds
    - tol: convergence tolerance on absolute change

    Returns a new dict with updated CO₂ values (does not modify inputs).
    """
    if rules_map is None:
        rules_map = rules

    # Work on copies and ensure floats
    updated = {s: float(v) for s, v in sector_data.items()}

    # Ensure all sectors mentioned in changes and rules exist in updated
    all_sectors = set(updated.keys()) | set(sector_changes.keys())
    for a, outs in (rules_map.items()):
        all_sectors.add(a)
        for b in outs.keys():
            all_sectors.add(b)
    for s in all_sectors:
        updated.setdefault(s, 0.0)

    # Apply direct changes (one-time), computing absolute change amounts
    # sector_changes are percentages in decimal form (e.g. -0.2)
    direct_delta: Dict[str, float] = {}
    for s, pct in sector_changes.items():
        try:
            p = float(pct)
        except Exception:
            p = 0.0
        base = updated.get(s, 0.0)
        delta = base * p
        direct_delta[s] = delta
        updated[s] = base + delta

    # Iteratively propagate effects: each change_amount on A causes change on B by
    # change_amount * coeff (coeff from rules_map[A][B]). We propagate until
    # changes become small or max_iterations reached.
    # Use damping to prevent unrealistic amplification
    current_delta = dict(direct_delta)  # latest changes to propagate
    damping_factor = 0.8  # reduce propagation each iteration

    for iteration in range(max_iterations):
        if not current_delta:
            break
        next_delta: Dict[str, float] = {}
        max_change = 0.0
        # For each sector that changed, propagate to its neighbours
        for a, change_amt in current_delta.items():
            if abs(change_amt) < tol:
                continue
            outs = rules_map.get(a, {})
            for b, coeff in outs.items():
                # effect on b is proportional to the absolute change in a
                effect = change_amt * float(coeff) * damping_factor
                if abs(effect) < tol:
                    continue
                updated[b] = updated.get(b, 0.0) + effect
                next_delta[b] = next_delta.get(b, 0.0) + effect
                max_change = max(max_change, abs(effect))

        # Check convergence
        if max_change <= tol:
            break
        current_delta = next_delta
        damping_factor *= 0.9  # further reduce on subsequent iterations

    return updated
