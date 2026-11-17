"""
features.py
-----------
Feature engineering for RUL prediction. Two groups:

  A. Summary features  - from the per-cycle summary table (preprocessed.parquet)
  B. Curve features    - from the per-cycle discharge curves in bat_dict
                         (Qdlin, Tdlin, discharge_dQdV)

Leakage rules enforced here:
  * a feature at cycle n uses only cycles <= n, plus an early-life reference
    (cycles REF_LO..REF_HI);
  * warm-up cycles (<= WARMUP) are dropped so the reference and the rolling
    windows are strictly in the past for every kept row;
  * `cycle` and `cycle_life` are NEVER used as features.

~42 features are built; notebook 04 then removes near-duplicates (|Spearman|
>= 0.95) on the TRAINING cells only, leaving > 20.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew

# Qdlin / dQdV are on a fixed linear voltage grid (dataset Vdlin: 2.0 -> 3.5 V,
# 1000 points). Not stored in the lean bat_dict, so reconstructed here.
VGRID = np.linspace(2.0, 3.5, 1000)
REF_LO, REF_HI = 2, 10        # early-life reference cycles
WARMUP = 20                   # drop cycles <= WARMUP (features need history)
PEAK_LO, PEAK_HI = 2.5, 3.4   # dQ/dV peak search window (volts)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _slope(y):
    """Least-squares slope of y against its position index."""
    n = len(y)
    if n < 2 or np.all(np.isnan(y)):
        return np.nan
    x = np.arange(n)
    return float(np.polyfit(x, y, 1)[0])


def _trapz(y, x):
    """Trapezoidal integral of y over x (version-independent)."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    return float(np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(x)))


def _valid_curve(c):
    """A cycle's curves are valid if Qdlin in (-0.1,1.3) and Tdlin in (20,60)."""
    q = np.asarray(c["Qdlin"], float)
    t = np.asarray(c["Tdlin"], float)
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(t))):
        return False
    return q.min() > -0.1 and q.max() < 1.3 and t.min() > 20 and t.max() < 60


# --------------------------------------------------------------------------
# A. Summary features
# --------------------------------------------------------------------------
def build_summary_features(df):
    df = df.sort_values(["cell", "cycle"]).copy()
    parts = []
    for cell, g in df.groupby("cell", sort=False):
        g = g.copy()
        ref = g[(g["cycle"] >= REF_LO) & (g["cycle"] <= REF_HI)]
        refQD = ref["QD"].mean()
        refIR = ref["IR"].mean()
        refTmax = ref["Tmax"].mean()
        refTavg = ref["Tavg"].mean()
        refTmin = ref["Tmin"].mean()
        refCT = ref["chargetime"].mean()
        ref_ce = (ref["QD"] / ref["QC"].replace(0, np.nan)).mean()

        # current-state
        g["QD_now"] = g["QD"]
        g["IR_now"] = g["IR"]
        g["Tavg_now"] = g["Tavg"]
        g["Tmin_now"] = g["Tmin"]
        g["Tmax_now"] = g["Tmax"]
        g["chargetime_now"] = g["chargetime"]
        g["T_range"] = g["Tmax"] - g["Tmin"]
        g["coulombic_eff"] = g["QD"] / g["QC"].replace(0, np.nan)

        # trend / rolling (trailing windows -> only past+current)
        g["QD_mean10"] = g["QD"].rolling(10, min_periods=3).mean()
        g["QD_std10"] = g["QD"].rolling(10, min_periods=3).std()
        g["QD_slope5"] = g["QD"].rolling(5, min_periods=3).apply(_slope, raw=True)
        g["QD_slope10"] = g["QD"].rolling(10, min_periods=3).apply(_slope, raw=True)
        g["QD_slope20"] = g["QD"].rolling(20, min_periods=5).apply(_slope, raw=True)
        g["QD_diff1"] = g["QD"].diff()
        g["QD_range10"] = (g["QD"].rolling(10, min_periods=3).max()
                           - g["QD"].rolling(10, min_periods=3).min())
        g["QD_slope_accel"] = g["QD_slope5"] - g["QD_slope20"]   # fade acceleration (knee)
        g["IR_mean10"] = g["IR"].rolling(10, min_periods=3).mean()
        g["IR_slope10"] = g["IR"].rolling(10, min_periods=3).apply(_slope, raw=True)
        g["IR_slope20"] = g["IR"].rolling(20, min_periods=5).apply(_slope, raw=True)
        g["IR_std10"] = g["IR"].rolling(10, min_periods=3).std()
        g["Tavg_mean10"] = g["Tavg"].rolling(10, min_periods=3).mean()
        g["Tavg_slope10"] = g["Tavg"].rolling(10, min_periods=3).apply(_slope, raw=True)
        g["chargetime_mean10"] = g["chargetime"].rolling(10, min_periods=3).mean()
        g["chargetime_slope10"] = g["chargetime"].rolling(10, min_periods=3).apply(_slope, raw=True)
        g["ce_slope10"] = g["coulombic_eff"].rolling(10, min_periods=3).apply(_slope, raw=True)
        g["ce_std10"] = g["coulombic_eff"].rolling(10, min_periods=3).std()

        # vs early-life reference
        g["cap_retention"] = g["QD"] / refQD
        g["cap_drop"] = refQD - g["QD"]
        g["IR_change"] = g["IR"] - refIR
        g["IR_ratio"] = g["IR"] / refIR if refIR else np.nan
        g["chargetime_change"] = g["chargetime"] - refCT
        g["chargetime_ratio"] = g["chargetime"] / refCT if refCT else np.nan
        g["Tmax_change"] = g["Tmax"] - refTmax
        g["Tavg_change"] = g["Tavg"] - refTavg
        g["Tmin_change"] = g["Tmin"] - refTmin
        g["ce_change"] = g["coulombic_eff"] - ref_ce
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# B. Discharge-curve features
# --------------------------------------------------------------------------
def build_curve_features(bat_dict):
    mask = (VGRID >= PEAK_LO) & (VGRID <= PEAK_HI)
    vpeak = VGRID[mask]
    dV = float(VGRID[1] - VGRID[0])
    i22 = int(np.argmin(np.abs(VGRID - 2.2)))
    i25 = int(np.argmin(np.abs(VGRID - 2.5)))
    i28 = int(np.argmin(np.abs(VGRID - 2.8)))
    i30 = int(np.argmin(np.abs(VGRID - 3.0)))
    i32 = int(np.argmin(np.abs(VGRID - 3.2)))

    rows = []
    for cell, d in bat_dict.items():
        cyc = d["cycles"]
        cyclenums = np.asarray(d["summary"]["cycle"])
        n = len(cyc)

        ref_idx = [j for j in range(n)
                   if REF_LO <= cyclenums[j] <= REF_HI and _valid_curve(cyc[str(j)])]
        if not ref_idx:
            continue
        refQ = np.nanmean([cyc[str(j)]["Qdlin"] for j in ref_idx], axis=0)
        refdq = np.abs(np.nanmean([cyc[str(j)]["dQdV"] for j in ref_idx], axis=0))
        ref_peak = float(np.nanmax(refdq[mask]))
        ref_peakV = float(vpeak[int(np.argmax(refdq[mask]))])

        last = None
        for j in range(n):
            c = cyc[str(j)]
            if _valid_curve(c):
                Q = np.asarray(c["Qdlin"], float)
                dq = np.abs(np.asarray(c["dQdV"], float))
                Td = np.asarray(c["Tdlin"], float)
                dQ = Q - refQ
                seg = dq[mask]
                pk = float(np.nanmax(seg))
                pkV = float(vpeak[int(np.argmax(seg))])
                feat = {
                    "dQ_mean": float(np.nanmean(dQ)), "dQ_min": float(np.nanmin(dQ)),
                    "dQ_max": float(np.nanmax(dQ)), "dQ_std": float(np.nanstd(dQ)),
                    "dQ_skew": float(skew(dQ, nan_policy="omit")),
                    "dQ_range": float(np.nanmax(dQ) - np.nanmin(dQ)),
                    "dQ_integral": float(_trapz(dQ, VGRID)),       # total capacity shift
                    "dQ_argminV": float(VGRID[int(np.nanargmin(dQ))]),  # voltage of worst fade
                    "dQ_V22": float(dQ[i22]), "dQ_V25": float(dQ[i25]),
                    "dQ_V28": float(dQ[i28]), "dQ_V30": float(dQ[i30]),
                    "dQ_V32": float(dQ[i32]),
                    "dqdv_peak_height": pk, "dqdv_peak_V": pkV,
                    "dqdv_V_shift": pkV - ref_peakV,
                    "dqdv_height_change": pk - ref_peak,
                    "dqdv_area": float(_trapz(seg, vpeak)),         # peak "mass"
                    "dqdv_fwhm": float((seg >= pk / 2).sum() * dV),   # peak width (broadening)
                    "Q_at_V30": float(Q[i30]),                        # absolute capacity at 3.0 V
                    "Td_max": float(np.nanmax(Td)), "Td_mean": float(np.nanmean(Td)),
                    "Td_range": float(np.nanmax(Td) - np.nanmin(Td)),
                }
                last = feat
            elif last is not None:
                feat = dict(last)          # fill invalid cycle from previous valid
            else:
                continue
            row = {"cell": cell, "cycle": int(cyclenums[j])}
            row.update(feat)
            rows.append(row)

    cf = pd.DataFrame(rows)
    # Td_mean_change vs early-life mean (first few valid discharge-temp means)
    cf["Td_mean_change"] = cf.groupby("cell")["Td_mean"].transform(
        lambda s: s - s.iloc[:5].mean())
    return cf


# --------------------------------------------------------------------------
# combine
# --------------------------------------------------------------------------
META = ["cell", "batch", "charge_policy", "cycle", "cycle_life", "RUL"]


def build_features(df, bat_dict, warmup=WARMUP):
    """Summary + curve features merged on (cell, cycle), warm-up dropped,
    returns a clean table with META columns + feature columns (no NaN)."""
    df = df.copy()
    df["cycle"] = df["cycle"].astype(int)

    sf = build_summary_features(df)
    cf = build_curve_features(bat_dict)
    cf["cycle"] = cf["cycle"].astype(int)

    feats = sf.merge(cf, on=["cell", "cycle"], how="left")
    curve_cols = [c for c in cf.columns if c not in ("cell", "cycle")]
    feats[curve_cols] = feats.groupby("cell")[curve_cols].ffill()

    feats = feats[feats["cycle"] > warmup].copy()
    feats = feats.dropna().reset_index(drop=True)
    return feats


def feature_columns(feats):
    """Names of the engineered feature columns (everything except META/raw)."""
    raw = {"QD", "QC", "IR", "Tavg", "Tmin", "Tmax", "chargetime"}
    return [c for c in feats.columns if c not in META and c not in raw]
