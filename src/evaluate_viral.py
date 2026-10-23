import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader

from graph_dataset import AffinityDataset, load_graphs
from affinity_model import AffinityModel


def main():
    print("=" * 55)
    print("STEP 7: VIRAL KI MODEL EVALUATION")
    print("=" * 55)

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "processed"
    checkpoint_path = root / "checkpoints" / "viral_ki_best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("\n[10%] Loading checkpoint and datasets...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )

    label_mean = float(checkpoint["label_mean"])

    graphs = load_graphs()
    test_ds = AffinityDataset(
        str(data_dir / "test_viral_graph.csv"),
        mtype="Ki",
        graphs=graphs
    )

    train_df = pd.read_csv(data_dir / "train_viral_graph.csv")
    train_df = train_df[
        train_df["mtype"].astype(str).str.lower() == "ki"
    ]
    train_mean = float(train_df["label"].mean())

    loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    print(f"Test examples: {len(test_ds):,}")
    print("\n[30%] Loading trained model...")

    model = AffinityModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    actual_all = []
    predicted_all = []

    print("\n[50%] Generating test predictions...")

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch).view(-1) + label_mean

            actual_all.extend(batch.y.view(-1).cpu().numpy().tolist())
            predicted_all.extend(predictions.cpu().numpy().tolist())

    actual = np.asarray(actual_all, dtype=float)
    predicted = np.asarray(predicted_all, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        raise RuntimeError("No valid test predictions were produced.")

    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    pearson = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if len(actual) > 1 and np.std(actual) > 0 and np.std(predicted) > 0
        else float("nan")
    )

    actual_rank = pd.Series(actual).rank().to_numpy()
    predicted_rank = pd.Series(predicted).rank().to_numpy()
    spearman = (
        float(np.corrcoef(actual_rank, predicted_rank)[0, 1])
        if len(actual) > 1
        and np.std(actual_rank) > 0
        and np.std(predicted_rank) > 0
        else float("nan")
    )

    baseline_rmse = float(np.sqrt(np.mean((actual - train_mean) ** 2)))
    baseline_mae = float(np.mean(np.abs(actual - train_mean)))

    print("\n[80%] RESULTS")
    print("-" * 45)
    print(f"Valid test samples:       {len(actual):,}")
    print(f"Model MAE:                {mae:.4f}")
    print(f"Model RMSE:               {rmse:.4f}")
    print(f"Pearson correlation:      {pearson:.4f}")
    print(f"Spearman correlation:     {spearman:.4f}")
    print(f"Mean baseline MAE:        {baseline_mae:.4f}")
    print(f"Mean baseline RMSE:       {baseline_rmse:.4f}")

    print("\n[90%] Saving individual predictions...")

    output = pd.DataFrame({
        "actual_pAff": actual,
        "predicted_pAff": predicted,
        "absolute_error": np.abs(errors)
    })

    output_path = root / "viral_ki_test_predictions.csv"
    output.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")
    print("\n[100%] Evaluation complete.")


if __name__ == "__main__":
    main()
