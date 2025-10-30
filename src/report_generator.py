import json
import pandas as pd
import os
from typing import Dict, Any, Union


def generate_summary(before_data: Union[Dict[str, float], pd.DataFrame], 
                     after_data: Union[Dict[str, float], pd.DataFrame] = None) -> Union[str, Dict[str, Any]]:
    """Generate a text summary comparing before/after CO₂ data or a dict summary from a DataFrame.
    
    If after_data is provided (dict or DataFrame), returns a formatted text summary string.
    If only before_data is provided as a DataFrame with 'baseline' and 'simulated' columns,
    returns a dict summary (backward compatibility).
    """
    # Backward compatibility: if before_data is a DataFrame with baseline/simulated, return dict
    if after_data is None and isinstance(before_data, pd.DataFrame):
        df = before_data
        if 'baseline' in df.columns and 'simulated' in df.columns:
            total_baseline = float(df['baseline'].sum())
            total_simulated = float(df['simulated'].sum())
            total_delta = total_baseline - total_simulated
            pct_reduction = (total_delta / total_baseline * 100.0) if total_baseline > 0 else 0.0
            per_sector = df.reset_index().to_dict(orient='records')
            return {
                'total_baseline': total_baseline,
                'total_simulated': total_simulated,
                'total_delta': total_delta,
                'pct_reduction': pct_reduction,
                'per_sector': per_sector
            }
    
    # New functionality: generate text summary from before/after dicts or DataFrames
    # Convert inputs to dicts if needed
    if isinstance(before_data, pd.DataFrame):
        # assume single column or index contains sectors
        if before_data.shape[1] == 1:
            before_dict = before_data.iloc[:, 0].to_dict()
        else:
            before_dict = before_data.to_dict(orient='index')
            # flatten if nested
            before_dict = {k: list(v.values())[0] if isinstance(v, dict) else v for k, v in before_dict.items()}
    else:
        before_dict = dict(before_data) if before_data else {}
    
    if isinstance(after_data, pd.DataFrame):
        if after_data.shape[1] == 1:
            after_dict = after_data.iloc[:, 0].to_dict()
        else:
            after_dict = after_data.to_dict(orient='index')
            after_dict = {k: list(v.values())[0] if isinstance(v, dict) else v for k, v in after_dict.items()}
    else:
        after_dict = dict(after_data) if after_data else {}
    
    # Calculate totals
    total_before = sum(float(v) for v in before_dict.values())
    total_after = sum(float(v) for v in after_dict.values())
    total_delta = total_before - total_after
    overall_pct = (total_delta / total_before * 100.0) if total_before > 0 else 0.0
    
    # Calculate per-sector changes
    sector_changes = {}
    for sector in before_dict.keys():
        b = float(before_dict.get(sector, 0.0))
        a = float(after_dict.get(sector, 0.0))
        delta = b - a
        pct = (delta / b * 100.0) if b > 0 else 0.0
        sector_changes[sector] = {'delta': delta, 'pct': pct}
    
    # Find sector with highest reduction (most positive delta)
    best_sector = None
    best_pct = 0.0
    for sector, info in sector_changes.items():
        if info['pct'] > best_pct:
            best_pct = info['pct']
            best_sector = sector
    
    # Build narrative summary
    if overall_pct > 0:
        direction = "reduced"
    elif overall_pct < 0:
        direction = "increased"
    else:
        direction = "had no change in"
    
    summary_text = (
        f"Overall CO₂ emissions {direction} by {abs(overall_pct):.1f}% "
        f"(from {total_before:.0f} to {total_after:.0f}).\n"
    )
    
    if best_sector and best_pct > 0:
        summary_text += (
            f"The most responsive sector was {best_sector.capitalize()}, "
            f"showing a {best_pct:.1f}% reduction."
        )
    else:
        summary_text += "No significant reductions were observed in individual sectors."
    
    return summary_text


def save_summary_json(summary: Dict[str, Any], filename: str, out_dir: str = None) -> str:
    if out_dir is None:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        out_dir = os.path.join(base, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    return path


def save_summary_csv(df: pd.DataFrame, filename: str, out_dir: str = None) -> str:
    if out_dir is None:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        out_dir = os.path.join(base, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    df.to_csv(path)
    return path
