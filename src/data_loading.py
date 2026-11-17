"""
data_loading.py
---------------
MATR / Severson battery dataset (3 batches) loading + merging.

  1. Download the three .mat struct files from data.matr.io (see download_info()).
  2. load_batch(mat_path, prefix) -> dict of cells for one batch.
  3. merge_batches(b1, b2, b3)    -> single bat_dict of 124 cells, with the
     batch1<->batch2 continuation cells joined and known bad cells removed.

Reference logic: rdbraatz/data-driven-prediction-of-battery-cycle-life-
before-capacity-degradation. Reproduced in modern h5py ([()] not .value).
This module only LOADS and MERGES. Checks/RUL live in preprocessing.py.
"""

import numpy as np
import h5py

DOWNLOAD = {
    # Batch 3 (2018-04-12) URL confirmed from the project page:
    "batch3": "https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download",
    # Batch 1 (2017-05-12) & Batch 2 (2017-06-30): open the project page
    #   https://data.matr.io/1/projects/5c48dd2bc625d700019f3204
    # click each batch, right-click the download button -> "Copy link address".
    "batch1": "PASTE_BATCH1_DOWNLOAD_URL_HERE",
    "batch2": "PASTE_BATCH2_DOWNLOAD_URL_HERE",
}

MAT_FILENAMES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}


def download_info():
    print("Run in a Colab cell (after filling batch1/batch2 URLs):\n")
    for b in ("batch1", "batch2", "batch3"):
        print(f"!wget -O {MAT_FILENAMES[b]} '{DOWNLOAD[b]}'")


def load_batch(mat_path, prefix):
    """Load one batch .mat (MATLAB v7.3 / HDF5) into a dict keyed b?c0, b?c1 ..."""
    f = h5py.File(mat_path, "r")
    batch = f["batch"]
    num_cells = batch["summary"].shape[0]
    bat_dict = {}

    for i in range(num_cells):
        cl = f[batch["cycle_life"][i, 0]][()]
        policy = f[batch["policy_readable"][i, 0]][()].tobytes()[::2].decode()

        def col(field):
            return np.hstack(f[batch["summary"][i, 0]][field][0, :].tolist())

        summary = {
            "IR": col("IR"), "QC": col("QCharge"), "QD": col("QDischarge"),
            "Tavg": col("Tavg"), "Tmin": col("Tmin"), "Tmax": col("Tmax"),
            "chargetime": col("chargetime"), "cycle": col("cycle"),
        }

        cycles_ref = f[batch["cycles"][i, 0]]
        cycle_dict = {}
        for j in range(cycles_ref["I"].shape[0]):
            cd = {}
            for name, key in [("I", "I"), ("Qc", "Qc"), ("Qd", "Qd"),
                              ("Qdlin", "Qdlin"), ("T", "T"), ("Tdlin", "Tdlin"),
                              ("V", "V"), ("dQdV", "discharge_dQdV"), ("t", "t")]:
                cd[name] = np.hstack(f[cycles_ref[key][j, 0]][()])
            cycle_dict[str(j)] = cd

        bat_dict[f"{prefix}c{i}"] = {
            "cycle_life": cl, "charge_policy": policy,
            "summary": summary, "cycles": cycle_dict,
        }
    return bat_dict


# Canonical removals / merges -------------------------------------------------
BATCH1_REMOVE = ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"]       # never reach 80%
CONT_BATCH1 = ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"]             # continued in batch2
CONT_BATCH2 = ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"]
CONT_ADDLEN = [662, 981, 1060, 208, 482]
BATCH3_REMOVE = ["b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"]   # noisy


def merge_batches(batch1, batch2, batch3):
    """Removals + batch1/batch2 continuation join -> 124 cells (41+43+40)."""
    batch1, batch2, batch3 = dict(batch1), dict(batch2), dict(batch3)

    for k in BATCH1_REMOVE:
        batch1.pop(k, None)

    for b1k, b2k, add in zip(CONT_BATCH1, CONT_BATCH2, CONT_ADDLEN):
        batch1[b1k]["cycle_life"] = batch1[b1k]["cycle_life"] + add
        for field in batch1[b1k]["summary"].keys():
            if field == "cycle":
                offset = len(batch1[b1k]["summary"][field])
                batch1[b1k]["summary"][field] = np.hstack(
                    (batch1[b1k]["summary"][field], batch2[b2k]["summary"][field] + offset))
            else:
                batch1[b1k]["summary"][field] = np.hstack(
                    (batch1[b1k]["summary"][field], batch2[b2k]["summary"][field]))
        last = len(batch1[b1k]["cycles"].keys())
        for j, jk in enumerate(batch2[b2k]["cycles"].keys()):
            batch1[b1k]["cycles"][str(last + j)] = batch2[b2k]["cycles"][jk]

    for k in CONT_BATCH2:
        batch2.pop(k, None)
    for k in BATCH3_REMOVE:
        batch3.pop(k, None)

    return {**batch1, **batch2, **batch3}


def batch_counts(batch1, batch2, batch3):
    b1 = len(batch1) - len(BATCH1_REMOVE)
    b2 = len(batch2) - len(CONT_BATCH2)
    b3 = len(batch3) - len(BATCH3_REMOVE)
    return {"batch1": b1, "batch2": b2, "batch3": b3, "total": b1 + b2 + b3}
