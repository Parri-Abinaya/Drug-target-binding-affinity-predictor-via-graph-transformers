import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("..\\results", exist_ok=True)

df = pd.read_csv("..\\all_model_comparison.csv")
metrics = [
    ("MAE", "Mean Absolute Error (lower is better)"),
    ("RMSE", "Root Mean Squared Error (lower is better)"),
    ("Pearson", "Pearson Correlation (higher is better)"),
    ("Spearman", "Spearman Correlation (higher is better)"),
]

for column, title in metrics:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(df["Model"], df[column])
    ax.set_title(title)
    ax.set_ylabel(column)
    ax.tick_params(axis="x", labelrotation=18)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output = os.path.join("..", "results", f"{column.lower()}_comparison.png")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", os.path.abspath(output))

# Combined summary table as a figure.
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.axis("off")
display_df = df.copy()
for col in ["MAE", "RMSE", "Pearson", "Spearman"]:
    display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
table = ax.table(
    cellText=display_df.values,
    colLabels=display_df.columns,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)
ax.set_title("Viral Ki Affinity Prediction: Test-Set Model Comparison", pad=20)
fig.tight_layout()
output = os.path.join("..", "results", "model_comparison_table.png")
fig.savefig(output, dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved:", os.path.abspath(output))
