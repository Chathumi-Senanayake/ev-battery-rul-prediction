"""
preprocessing.py
----------------
Merged bat_dict (124 cells) -> clean per-cycle table + RUL label + quality
checks. Also the EOL-reached filter used in notebook 02. No feature engineering.

Data point = one (cell, cycle). Label = RUL = cycle_life - cycle.
End of life = discharge capacity falls to 0.88 Ah (80% of 1.1 Ah nominal).
"""

import numpy as np
import pandas as pd

NOMINAL_AH = 1.1
EOL_AH = 0.88
SUMMARY_FIELDS = ["IR", "QC", "QD", "Tavg", "Tmin", "Tmax", "chargetime", "cycle"]

VALID_RANGES = {
    "QD": (0.0, 1.3), "QC": (0.0, 1.3), "IR": (0.0, 0.1),
    "Tavg": (20.0, 60.0), "Tmin": (20.0, 60.0), "Tmax": (20.0, 60.0),
    "chargetime": (0.0, 60.0),
}


def build_summary_table(bat_dict):
    """Flatten bat_dict into a long DataFrame: one row per (cell, cycle)."""
    rows = []
    for cell, d in bat_dict.items():
        s = d["summary"]
        cl = float(np.ravel(d["cycle_life"])[0])
        batch = cell[:2]
        for k in range(len(s["cycle"])):
            row = {"cell": cell, "batch": batch, "cycle_life": cl,
                   "charge_policy": d["charge_policy"]}
            for f in SUMMARY_FIELDS:
                row[f] = float(s[f][k])
            rows.append(row)
    return pd.DataFrame(rows)


def add_rul(df):
    df = df.copy()
    df["RUL"] = (df["cycle_life"] - df["cycle"]).clip(lower=0)
    return df


def reached_eol(bat_dict):
    """Cells whose min QD dropped to EOL (sanity: most should be True)."""
    out = {}
    for cell, d in bat_dict.items():
        out[cell] = bool(np.nanmin(d["summary"]["QD"]) <= EOL_AH)
    return out


def quality_report(df):
    rep = {"n_rows": len(df), "n_cells": df["cell"].nunique()}
    rep["missing_per_col"] = df.isna().sum().to_dict()
    rep["duplicate_cell_cycle"] = int(df.duplicated(subset=["cell", "cycle"]).sum())
    bad_order = [c for c, g in df.groupby("cell") if np.any(np.diff(g["cycle"].values) <= 0)]
    rep["cells_with_bad_cycle_order"] = bad_order
    rep["IR_le_0"] = int((df["IR"] <= 0).sum())
    rep["QD_le_0"] = int((df["QD"] <= 0).sum())
    rep["out_of_range"] = {c: int(((df[c] < lo) | (df[c] > hi)).sum())
                           for c, (lo, hi) in VALID_RANGES.items() if c in df}
    rep["lifetime_stats"] = df.groupby("cell")["cycle_life"].first().describe().to_dict()
    return rep


def print_report(rep):
    print(f"rows: {rep['n_rows']}   cells: {rep['n_cells']}")
    print(f"duplicate (cell,cycle): {rep['duplicate_cell_cycle']}")
    print(f"bad cycle order cells : {rep['cells_with_bad_cycle_order'] or 'none'}")
    print(f"IR<=0: {rep['IR_le_0']}   QD<=0: {rep['QD_le_0']}")
    print(f"out of range: {rep['out_of_range']}")
    miss = {k: v for k, v in rep["missing_per_col"].items() if v}
    print(f"missing (non-zero): {miss or 'none'}")


def basic_clean(df, drop_first_cycle=True):
    """Safe-only cleaning: drop dup (cell,cycle), optional first cycle, sort."""
    df = df.drop_duplicates(subset=["cell", "cycle"]).copy()
    if drop_first_cycle:
        df = df[df["cycle"] > 1]
    return df.sort_values(["cell", "cycle"]).reset_index(drop=True)
