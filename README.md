# House Prices Prediction with TabNet & Optuna

End-to-end house price prediction on the **Ames Housing** dataset using a deep tabular model (TabNet) with systematic hyperparameter optimization via Optuna.  
The project demonstrates how careful modeling, evaluation, and tuning can lead to **dramatic performance improvements** over strong baselines.

------------

## Project Overview

- **Task:** Regression — predict house sale prices
- **Dataset:** Kaggle *House Prices: Advanced Regression Techniques*
- **Target:** `SalePrice` (log-transformed during training)
- **Model:** TabNet Regressor (PyTorch)
- **Optimization:** Optuna (random + pruned hyperparameter search)
- **Evaluation:** Log-space + real USD-space metrics

-------------------------

## Key Techniques Used

- Robust preprocessing with mixed numerical & categorical features
- Log-transform of target variable to stabilize variance
- One-hot encoding + imputation pipeline
- Deep tabular learning with **TabNet**
- GPU-accelerated training (CUDA)
- **Optuna hyperparameter optimization with pruning**
- Early stopping to prevent overfitting
- Evaluation in **real USD space** (business-relevant metrics)



  Results

 Baseline vs  Optimized Model

| Metric | Initial TabNet (Baseline) | Optuna-Tuned TabNet | Improvement |
|------|---------------------------|---------------------|-------------|
| RMSE (log) | ~0.243 | **0.136** | ↓ ~44% |
| RMSE (USD) | ~$41,906 | **$25,192** | ↓ ~40% |
| MAE (USD) | ~$29,616 | **$16,426** | ↓ ~45% |
| MAPE | ~18.83% | **9.86%** | ↓ ~48% |

 **Overall:** ~40–45% reduction in prediction error after systematic tuning.

------

##  Interpretation

- The baseline TabNet already matched strong linear models (Ridge/Lasso).
- Hyperparameter optimization unlocked TabNet’s ability to model **non-linear feature interactions**.
- Final performance approaches **top-tier Kaggle solutions**, without heavy manual feature engineering.
- Error rates below **10% MAPE** are strong for real-world housing price prediction.

-----

##  Evaluation Metrics Explained

- **RMSE (USD):** Penalizes large errors; reflects worst-case risk.
- **MAE (USD):** Average absolute error; business-friendly metric.
- **MAPE:** Mean percentage error; intuitive measure of accuracy.
- **Tolerance-based accuracy** (±10%) used internally for interpretability.

> Note: Classification metrics (Accuracy, F1) are not applicable to regression and were intentionally avoided.

----

## Hyperparameter Optimization

- Search strategy: Optuna + Median Pruner
- Parameters tuned include:
  - `n_d`, `n_a`, `n_steps`
  - `gamma`, `lambda_sparse`
  - learning rate, batch size
  - mask type (`entmax`, `sparsemax`)
- Poor-performing trials were automatically stopped early to save compute.

All trial results are saved in:
