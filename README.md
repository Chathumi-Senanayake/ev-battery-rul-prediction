# EV Battery RUL Prediction (EC5203)

Machine Learning–based Remaining Useful Life (RUL) prediction for EV
lithium-ion batteries. Dataset: **MATR / Severson** (A123 APR18650M1A LFP cells),
all **3 batches** → 124 cells after the canonical merge.

- **Task:** supervised regression (predict RUL in cycles)
- **Data point:** one (cell, cycle); **Label:** `RUL = cycle_life − cycle`
- **End of life:** discharge capacity ≤ 0.88 Ah (80% of 1.1 Ah nominal)
- **No neural networks** (course rule). 5 classical models.
- **Members (proposal order):** Senanayake, Wijayagurusinghe, Deepaka, Nimnadi

---

## Notebook structure

| # | Notebook | Does |
|---|---|---|
| 01 | `01_data_loading_checks` | Read 3 batches, missing / duplicate / invalid checks |
| 02 | `02_preprocessing_rul` | Remove cells that never reach EOL, build RUL label |
| 03 | `03_feature_engineering` | Summary features + discharge-curve features |
| 04 | `04_selection_and_split` | Drop similar features, battery-wise train/test split |
| 05 | `05_linear_regression` | Model 1 |
| 06 | `06_random_forest` | Model 2 |
| 07 | `07_svr` | Model 3 |
| 08 | `08_gradient_boosting` | Model 4 |
| 09 | `09_xgboost` | Model 5 |
| 10 | `10_comparison` | Final test, model comparison, plots, feature importance |

## Work division

| Member | Notebooks | Branches | Report / docs |
|---|---|---|---|
| **Senanayake** | 01 + 05 (Linear Regression) | `data/01-loading-checks`, `model/linear-regression` | Intro, dataset & cleaning. GitHub README |
| **Wijayagurusinghe** | 02 + 04 + 06 (Random Forest) | `data/02-preprocessing-rul`, `data/04-selection-split`, `model/random-forest` | Methodology (RUL, split, metrics) |
| **Deepaka** | 03 (heaviest) + 07 (SVR) | `data/03-features`, `model/svr` | Features section. Proposal update |
| **Nimnadi** | 10 + 08 (Gradient Boosting) + 09 (XGBoost) | `compare/10-comparison`, `model/gradient-boosting`, `model/xgboost` | Results, discussion, limitations. Slides |

> Gradient Boosting and XGBoost are close siblings, so they sit with one person.
> Swap names/assignments freely.

---

## File structure

```
ev-battery-rul-prediction/
├── notebooks/
│   ├── 01_data_loading_checks.ipynb     # Senanayake        [shared foundation]
│   ├── 02_preprocessing_rul.ipynb       # Wijayagurusinghe
│   ├── 03_feature_engineering.ipynb     # Deepaka
│   ├── 04_selection_and_split.ipynb     # Wijayagurusinghe
│   ├── 05_linear_regression.ipynb       # Senanayake
│   ├── 06_random_forest.ipynb           # Wijayagurusinghe
│   ├── 07_svr.ipynb                     # Deepaka
│   ├── 08_gradient_boosting.ipynb       # Nimnadi
│   ├── 09_xgboost.ipynb                 # Nimnadi
│   └── 10_comparison.ipynb              # Nimnadi
├── src/
│   ├── data_loading.py                  # load_batch(), merge_batches()
│   └── preprocessing.py                 # build_summary_table(), add_rul(), checks
├── data/                                # .mat / .pkl / .parquet  (gitignored)
├── outputs/                             # metrics, figures
├── requirements.txt
├── setup_branches.sh
└── README.md
```

Pipeline order: `01 → 02 → 03 → 04 → (05..09 per member) → 10`.
Notebooks `01`–`04` save intermediate files into `DATA_DIR`
(`bat_dict.pkl`, `preprocessed.parquet`, `features.parquet`, `train.parquet`,
`test.parquet`), so model notebooks just load the split and train.

---

## Colab workflow

Every notebook starts with the same setup cell (clone/pull → pip install →
mount Drive). Big data files live in `DATA_DIR` on Drive so they persist.

> **Data note:** Batch 3's download URL is filled in `src/data_loading.py`;
> paste Batch 1 & 2 URLs there (project page → each batch → right-click
> download → Copy link).

## Git workflow

Each member works on their own branch and opens a PR into `main`:

```bash
git checkout main && git pull
git checkout -b model/random-forest
# ...work in your notebook...
git add notebooks/06_random_forest.ipynb
git commit -m "random forest: baseline + CV"
git push -u origin model/random-forest
```

Run `./setup_branches.sh` once (anyone with push access) to create all 10 branches.
