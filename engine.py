"""MatEIS calculation engine: database, validation, filtering, Ashby indices, Pareto."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "materials.json"
FAMILIES = ["Metal", "Ceramic", "Polymer", "Composite"]
PROPS = {  # key: (label, unit, preferred direction: +1 higher is better, -1 lower)
    "density": ("Density", "kg/m³", -1), "E": ("Young's modulus", "GPa", 1),
    "sigma_y": ("Yield strength", "MPa", 1), "cost": ("Cost", "USD/kg", -1)}
# Material indices M = prop**exp / (rho [* cost]); (prop, exp, include_cost)
INDICES = {
    "Light, strong tie  (σy/ρ)": ("sigma_y", 1.0, False),
    "Light, stiff tie  (E/ρ)": ("E", 1.0, False),
    "Light, stiff beam  (E^½/ρ)": ("E", 0.5, False),
    "Light, stiff plate  (E^⅓/ρ)": ("E", 1 / 3, False),
    "Light, strong beam  (σy^⅔/ρ)": ("sigma_y", 2 / 3, False),
    "Cheap, stiff beam  (E^½/ρC)": ("E", 0.5, True),
    "Cheap, strong tie  (σy/ρC)": ("sigma_y", 1.0, True)}


def load(path=DB_PATH) -> pd.DataFrame:
    return pd.DataFrame(json.loads(Path(path).read_text(encoding="utf-8")))


def validate(row: dict, existing=()) -> dict:
    """Return a cleaned row or raise ValueError with a readable message."""
    name = str(row.get("material", "")).strip()
    if not name:
        raise ValueError("Material name is required.")
    if name.lower() in {e.lower() for e in existing}:
        raise ValueError(f"'{name}' already exists.")
    if row.get("family") not in FAMILIES:
        raise ValueError(f"Family must be one of {FAMILIES}.")
    out = {"material": name, "family": row["family"]}
    for k, (label, unit, _) in PROPS.items():
        try:
            v = float(row[k])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{label} must be a number.")
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"{label} must be a positive, finite number.")
        out[k] = v
    return out


def add_material(row: dict, path=DB_PATH) -> None:
    df = load(path)
    clean = validate(row, df["material"])
    data = df.to_dict("records") + [clean]
    Path(path).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def filter_materials(df, limits=None, families=None) -> pd.DataFrame:
    """limits: {prop: (min, max)}, bounds inclusive; None means unbounded."""
    mask = pd.Series(True, index=df.index)
    if families is not None:
        mask &= df["family"].isin(families)
    for p, (lo, hi) in (limits or {}).items():
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"{p}: minimum exceeds maximum.")
        if lo is not None:
            mask &= df[p] >= lo
        if hi is not None:
            mask &= df[p] <= hi
    return df[mask]


def index_value(df, key) -> pd.Series:
    prop, exp, with_cost = INDICES[key]
    return df[prop] ** exp / (df["density"] * (df["cost"] if with_cost else 1))


def rank(df, key) -> pd.DataFrame:
    out = df.assign(index=index_value(df, key)).sort_values("index", ascending=False)
    out["relative_to_best_%"] = 100 * out["index"] / out["index"].max() if len(out) else []
    return out.reset_index(drop=True)


def pareto_front(df, x, y) -> pd.DataFrame:
    """Non-dominated materials for axes x, y, using PROPS preferred directions."""
    pts = df[[x, y]].to_numpy(float) * np.array([PROPS[x][2], PROPS[y][2]])
    keep = [not any((q >= p).all() and (q > p).any() for q in pts) for p in pts]
    return df[keep].sort_values(x)
