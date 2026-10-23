import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_DIR = ".."

FILES = {
    "Baseline": ("viral_ki_test_predictions.csv", "predicted_pAff"),
    "Full multimodal": ("viral_ki_multimodal_test_predictions.csv", "multimodal_predicted_pAff"),
    "No graph cross-attention (C)": ("viral_ki_no_graph_cross_predictions.csv", "predicted_pAff"),
    "No SMILES cross-attention (D)": ("viral_ki_no_smiles_cross_predictions.csv", "predicted_pAff"),
}

def get_actual_column(df):
    for col in ["actual_pAff", "actual", "y_true", "true", "actual_y"]:
        if col in df.columns:
            return col
    raise ValueError(f"No actual-label column found: {list(df.columns)}")

def calculate_metrics(actual, predicted):
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]

    if len(actual) < 2:
        raise ValueError("Fewer than two valid predictions.")

    return {
        "N": len(actual),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "Pearson": pearsonr(actual, predicted).statistic,
        "Spearman": spearmanr(actual, predicted).statistic,
    }

results = []
reference_actual = None

for model_name, (filename, pred_col) in FILES.items():
    path = os.path.join(PROJECT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing prediction file: {path}")

    df = pd.read_csv(path)
    actual_col = get_actual_column(df)

    if pred_col not in df.columns:
        raise ValueError(
            f"Expected prediction column '{pred_col}' in {filename}. "
            f"Available columns: {list(df.columns)}"
        )

    actual = df[actual_col].to_numpy(dtype=float)
    predicted = df[pred_col].to_numpy(dtype=float)

    if reference_actual is None:
        reference_actual = actual.copy()
    elif len(actual) != len(reference_actual) or not np.allclose(
        actual, reference_actual, equal_nan=True
    ):
        raise ValueError(
            f"Actual labels in {filename} do not match the baseline file "
            "in the same order. Do not compare until alignment is fixed."
        )

    metrics = calculate_metrics(actual, predicted)
    metrics["Model"] = model_name
    results.append(metrics)

results_df = pd.DataFrame(results)[
    ["Model", "N", "MAE", "RMSE", "Pearson", "Spearman"]
]

baseline_df = pd.read_csv(
    os.path.join(PROJECT_DIR, FILES["Baseline"][0])
)
actual = baseline_df[get_actual_column(baseline_df)].to_numpy(dtype=float)
mean_pred = np.full(len(actual), np.nanmean(actual))
mean_metrics = calculate_metrics(actual, mean_pred)

print("\n=== FOUR-MODEL TEST COMPARISON ===")
print(results_df.round(4).to_string(index=False))
print(f"\nMean-label baseline MAE:  {mean_metrics['MAE']:.4f}")
print(f"Mean-label baseline RMSE: {mean_metrics['RMSE']:.4f}")

out_path = os.path.join(PROJECT_DIR, "all_model_comparison.csv")
results_df.to_csv(out_path, index=False)
print("\nSaved metrics to:", os.path.abspath(out_path))


