import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader

from graph_dataset import AffinityDataset, load_graphs
from affinity_model_multimodal import MultimodalAffinityModel


def metrics(actual, predicted):
    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    pearson = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if len(actual) > 1
        and np.std(actual) > 0
        and np.std(predicted) > 0
        else float("nan")
    )

    ar = pd.Series(actual).rank().to_numpy()
    pr = pd.Series(predicted).rank().to_numpy()
    spearman = (
        float(np.corrcoef(ar, pr)[0, 1])
        if len(actual) > 1
        and np.std(ar) > 0
        and np.std(pr) > 0
        else float("nan")
    )
    return mae, rmse, pearson, spearman


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "processed"
    checkpoint_path = root / "checkpoints" / "viral_ki_multimodal_best.pt"
    baseline_path = root / "viral_ki_test_predictions.csv"

    for path in (checkpoint_path, baseline_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    print("Loading multimodal checkpoint...", flush=True)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model = MultimodalAffinityModel(**checkpoint.get("config", {})).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    label_mean = float(checkpoint["label_mean"])

    print("Loading molecular graphs...", flush=True)
    graphs = load_graphs()
    test_ds = AffinityDataset(
        str(data_dir / "test_viral_graph.csv"),
        mtype="Ki",
        graphs=graphs,
    )
    loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    actual_all, predicted_all = [], []
    print(f"Generating predictions for {len(test_ds)} test samples...", flush=True)

    with torch.no_grad():
        for i, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            pred = model(batch).view(-1) + label_mean
            actual_all.extend(batch.y.view(-1).cpu().numpy().tolist())
            predicted_all.extend(pred.cpu().numpy().tolist())

            if i % 10 == 0:
                print(f"  Batch {i}/{len(loader)}", flush=True)

    actual = np.asarray(actual_all, dtype=float)
    predicted = np.asarray(predicted_all, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]

    baseline = pd.read_csv(baseline_path)

    if len(baseline) != len(actual):
        raise RuntimeError(
            f"Sample-count mismatch: baseline has {len(baseline)}, "
            f"multimodal has {len(actual)}. Cannot compare safely."
        )

    if not np.allclose(
        baseline["actual_pAff"].to_numpy(dtype=float),
        actual,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise RuntimeError(
            "Test labels/order do not match the baseline CSV. "
            "Comparison stopped to avoid misleading results."
        )

    bpred = baseline["predicted_pAff"].to_numpy(dtype=float)
    bvalid = np.isfinite(bpred) & np.isfinite(actual)
    if not np.all(bvalid):
        raise RuntimeError("Baseline predictions contain invalid values.")

    mm_mae, mm_rmse, mm_p, mm_s = metrics(actual, predicted)
    b_mae, b_rmse, b_p, b_s = metrics(actual, bpred)

    train_df = pd.read_csv(data_dir / "train_viral_graph.csv")
    train_df = train_df[
        train_df["mtype"].astype(str).str.lower() == "ki"
    ]
    train_mean = float(train_df["label"].mean())
    mean_mae = float(np.mean(np.abs(actual - train_mean)))
    mean_rmse = float(np.sqrt(np.mean((actual - train_mean) ** 2)))

    print("\n=== FAIR TEST-SET COMPARISON ===")
    print(f"Matched test samples: {len(actual)}")
    print(f"{'Metric':<22}{'Baseline':>12}{'Multimodal':>14}")
    print("-" * 48)
    print(f"{'MAE':<22}{b_mae:>12.4f}{mm_mae:>14.4f}")
    print(f"{'RMSE':<22}{b_rmse:>12.4f}{mm_rmse:>14.4f}")
    print(f"{'Pearson':<22}{b_p:>12.4f}{mm_p:>14.4f}")
    print(f"{'Spearman':<22}{b_s:>12.4f}{mm_s:>14.4f}")
    print(f"{'Mean baseline MAE':<22}{mean_mae:>12.4f}")
    print(f"{'Mean baseline RMSE':<22}{mean_rmse:>12.4f}")

    if b_rmse > 0:
        improvement = 100 * (b_rmse - mm_rmse) / b_rmse
        print(f"\nRMSE change vs baseline: {improvement:+.2f}%")
        print("(Positive means lower RMSE; negative means worse RMSE.)")

    out = pd.DataFrame({
        "actual_pAff": actual,
        "baseline_predicted_pAff": bpred,
        "multimodal_predicted_pAff": predicted,
        "baseline_absolute_error": np.abs(bpred - actual),
        "multimodal_absolute_error": np.abs(predicted - actual),
    })
    out_path = root / "viral_ki_multimodal_test_predictions.csv"
    out.to_csv(out_path, index=False)
    print("\nSaved predictions:", out_path)


if __name__ == "__main__":
    main()
