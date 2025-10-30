"""Visualization helpers for Streamlit using Plotly.

Provides two main helpers:
- display_bar_chart(before_data, after_data): side-by-side bar chart
- display_heatmap(data): simple intensity heatmap by sector
"""

from typing import Dict, Any, Sequence
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def _to_series(data) -> pd.Series:
    """Convert input (dict/Series/DataFrame) to pd.Series indexed by sector."""
    if isinstance(data, pd.Series):
        return data.astype(float)
    if isinstance(data, pd.DataFrame):
        # if DataFrame has 'simulated' or 'value' pick a sensible column
        if 'simulated' in data.columns:
            s = data['simulated']
        elif 'value' in data.columns:
            s = data['value']
        elif data.shape[1] == 1:
            s = data.iloc[:, 0]
        else:
            # fallback: take first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                s = data[numeric_cols[0]]
            else:
                # create zero series
                s = pd.Series({c: 0.0 for c in data.index})
        return s.astype(float)
    if isinstance(data, dict):
        return pd.Series(data).astype(float)
    # attempt to coerce
    try:
        return pd.Series(dict(data)).astype(float)
    except Exception:
        return pd.Series(dtype=float)


def display_bar_chart(before_data, after_data, title: str = 'Before vs After CO₂ by sector') -> None:
    """Render a grouped bar chart in Streamlit comparing before vs after values.

    Inputs can be dicts, pandas Series, or DataFrames. Values will be aligned by
    sector (index) and missing values treated as 0.
    """
    before = _to_series(before_data)
    after = _to_series(after_data)

    # Align sectors
    all_index = sorted(set(before.index.tolist()) | set(after.index.tolist()))
    before = before.reindex(all_index, fill_value=0.0)
    after = after.reindex(all_index, fill_value=0.0)

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Before', x=all_index, y=before.values, marker_color='indianred'))
    fig.add_trace(go.Bar(name='After', x=all_index, y=after.values, marker_color='seagreen'))
    fig.update_layout(barmode='group', title=title, xaxis_title='Sector', yaxis_title='CO₂ Emissions')

    st.plotly_chart(fig, use_container_width=True)


def display_heatmap(data, title: str = 'Sector CO₂ Heatmap') -> None:
    """Display a simple intensity heatmap by sector.

    For simplicity this uses a single-row heatmap with sectors on the x-axis and
    intensity representing the CO₂ value. Users may expand to a correlation
    matrix later.
    """
    series = _to_series(data)
    if series.empty:
        st.info('No data available for heatmap')
        return

    sectors = list(series.index)
    values = series.values.astype(float)

    # Normalize values for color intensity (preserve absolute values in hover)
    # Create a 2D array with one row
    z = [values]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=sectors,
        y=['CO₂'],
        colorscale='YlOrRd',
        colorbar=dict(title='CO₂')
    ))
    fig.update_layout(title=title, xaxis_nticks=len(sectors))

    st.plotly_chart(fig, use_container_width=True)


# Compatibility wrappers (older API expected functions that return figure objects)
def bar_comparison(df, title: str = 'Baseline vs Simulated'):
    """Compatibility: return a Plotly Figure comparing baseline vs simulated.

    Accepts a DataFrame with columns 'baseline' and 'simulated' or dict-like inputs.
    """
    # try to extract baseline and simulated from df
    if hasattr(df, 'to_dict') and 'baseline' in getattr(df, 'columns', []):
        before = _to_series(df['baseline'])
        after = _to_series(df['simulated'])
    else:
        # fallback: if df is dict-like with nested values
        try:
            ser_before = _to_series({k: v.get('baseline', v) if isinstance(v, dict) else v for k, v in dict(df).items()})
            ser_after = _to_series({k: v.get('simulated', v) if isinstance(v, dict) else v for k, v in dict(df).items()})
            before = ser_before
            after = ser_after
        except Exception:
            before = _to_series(df)
            after = _to_series(df)

    all_index = sorted(set(before.index.tolist()) | set(after.index.tolist()))
    before = before.reindex(all_index, fill_value=0.0)
    after = after.reindex(all_index, fill_value=0.0)

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Baseline', x=all_index, y=before.values, marker_color='indianred'))
    fig.add_trace(go.Bar(name='Simulated', x=all_index, y=after.values, marker_color='seagreen'))
    fig.update_layout(barmode='group', title=title, xaxis_title='Sector', yaxis_title='CO₂ Emissions')
    return fig


def summary_table(df):
    """Compatibility: return a Plotly Table figure from a DataFrame."""
    import pandas as _pd
    # If df is not a DataFrame, attempt to coerce
    if not isinstance(df, _pd.DataFrame):
        try:
            df = _pd.DataFrame(df)
        except Exception:
            df = _pd.DataFrame()

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.reset_index().columns), fill_color='paleturquoise'),
        cells=dict(values=[df.reset_index()[c] for c in df.reset_index().columns])
    )])
    return fig

