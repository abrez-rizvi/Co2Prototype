import os
import json
import pandas as pd
from typing import Dict, Any, Tuple
from datetime import datetime


def _resolve_data_dir() -> str:
    base = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(base, '..', 'data'))


def _resolve_outputs_dir() -> str:
    base = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(base, '..', 'outputs'))


def _normalize_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize dataset to the requested format:
    {
      "city": "Delhi",
      "sectors": {"transport": 1200, "energy": 2200}
    }

    Accepts both the simple mapping and older formats where each sector
    may be an object with a 'baseline' key.
    """
    if not isinstance(dataset, dict):
        raise ValueError('dataset must be a dict')

    city = dataset.get('city') or dataset.get('name') or 'Unknown'
    sectors = dataset.get('sectors', {})

    normalized = {}
    for s, v in sectors.items():
        # if the value is a mapping with baseline key, extract it
        if isinstance(v, dict):
            if 'baseline' in v:
                normalized[s] = float(v['baseline'])
            else:
                # try to find a numeric value in dict
                # pick first numeric value if present
                num = None
                for vv in v.values():
                    try:
                        num = float(vv)
                        break
                    except Exception:
                        continue
                normalized[s] = float(num) if num is not None else 0.0
        else:
            # assume numeric
            try:
                normalized[s] = float(v)
            except Exception:
                normalized[s] = 0.0

    return {'city': city, 'sectors': normalized}


def load_presets() -> Dict[str, Dict[str, Any]]:
    """
    Loads all JSON files in the `data/` folder and returns a mapping
    filename (without .json) -> normalized dataset dict.
    """
    data_dir = _resolve_data_dir()
    presets: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(data_dir):
        return presets

    for fname in os.listdir(data_dir):
        if not fname.lower().endswith('.json'):
            continue
        path = os.path.join(data_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                payload = json.load(fh)
            normalized = _normalize_dataset(payload)
            presets[os.path.splitext(fname)[0]] = normalized
        except Exception:
            # skip invalid files
            continue
    return presets


def list_cities() -> list:
    """Return list of preset city names (file basenames without .json)."""
    presets = load_presets()
    return sorted(list(presets.keys()))


def load_city(city_name: str) -> Dict[str, Any]:
    """Load a city dataset by name from data/ and return normalized dict.

    If the file is not present, falls back to presets loaded by load_presets().
    """
    data_dir = _resolve_data_dir()
    path = os.path.join(data_dir, f"{city_name}.json")
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as fh:
            payload = json.load(fh)
        return _normalize_dataset(payload)
    # fallback to presets
    presets = load_presets()
    return presets.get(city_name, {})


def sectors_to_dataframe(city_data: Dict[str, Any]):
    """Convert city_data to a pandas DataFrame with index=sector and column 'baseline'."""
    sectors = city_data.get('sectors', {}) if isinstance(city_data, dict) else {}
    rows = []
    for name, val in sectors.items():
        try:
            baseline = float(val)
        except Exception:
            baseline = 0.0
        rows.append({'sector': name, 'baseline': baseline})
    import pandas as _pd
    if not rows:
        return _pd.DataFrame(columns=['baseline']).set_index(_pd.Index([], name='sector'))
    return _pd.DataFrame(rows).set_index('sector')


def load_custom_data(uploaded_file) -> Dict[str, Any]:
    """
    Parse an uploaded JSON file (e.g. Streamlit UploadedFile or file-like) and
    return a normalized dataset dict.

    uploaded_file: object with .read() returning bytes or a path-like string
    """
    raw = None
    # If it's a path-like string
    if isinstance(uploaded_file, str) and os.path.isfile(uploaded_file):
        with open(uploaded_file, 'r', encoding='utf-8') as fh:
            raw = fh.read()
    else:
        # try read() (Streamlit UploadedFile)
        if hasattr(uploaded_file, 'read'):
            content = uploaded_file.read()
            # bytes -> str
            if isinstance(content, (bytes, bytearray)):
                raw = content.decode('utf-8')
            else:
                raw = str(content)
        else:
            # last resort: treat as JSON string
            raw = str(uploaded_file)

    try:
        payload = json.loads(raw)
    except Exception as e:
        raise ValueError(f'Failed to parse uploaded JSON: {e}')

    return _normalize_dataset(payload)


def save_results(city: str, data: Any, out_dir: str = None) -> Tuple[str, str]:
    """
    Save simulation results for `city` into outputs folder as JSON and CSV.

    - `city`: name used in filenames
    - `data`: can be a dict (summary), list-of-dicts, or pandas.DataFrame

    Returns tuple (json_path, csv_path)
    """
    if out_dir is None:
        out_dir = _resolve_outputs_dir()
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    json_fname = f"{city}_results_{ts}.json"
    csv_fname = f"{city}_results_{ts}.csv"

    json_path = os.path.join(out_dir, json_fname)
    csv_path = os.path.join(out_dir, csv_fname)

    # Save JSON
    try:
        with open(json_path, 'w', encoding='utf-8') as fh:
            # if pandas DataFrame, convert to records
            if hasattr(data, 'to_dict') and not isinstance(data, dict):
                # DataFrame or similar
                json.dump(data.to_dict(orient='records'), fh, indent=2)
            else:
                json.dump(data, fh, indent=2)
    except Exception as e:
        raise IOError(f'Failed to save JSON results: {e}')

    # Save CSV: try to create a DataFrame
    try:
        if hasattr(data, 'to_csv'):
            # DataFrame
            data.to_csv(csv_path, index=True)
        else:
            # data could be dict-summary or list-of-dicts
            if isinstance(data, dict) and 'per_sector' in data and isinstance(data['per_sector'], list):
                df = pd.DataFrame(data['per_sector'])
                df.to_csv(csv_path, index=False)
            elif isinstance(data, dict):
                # convert mapping to two-column table if possible
                # e.g., {'sector': value}
                try:
                    df = pd.DataFrame([data])
                    df.to_csv(csv_path, index=False)
                except Exception:
                    # fallback: write an empty CSV
                    pd.DataFrame().to_csv(csv_path)
            elif isinstance(data, list):
                df = pd.DataFrame(data)
                df.to_csv(csv_path, index=False)
            else:
                # unknown shape
                pd.DataFrame().to_csv(csv_path)
    except Exception as e:
        raise IOError(f'Failed to save CSV results: {e}')

    return json_path, csv_path
