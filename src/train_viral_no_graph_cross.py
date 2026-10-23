"""Train and evaluate the model without graph cross-attention."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from graph_dataset import AffinityDataset, load_graphs
from affinity_model_no_graph_cross import AblationNoGraphCross


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "processed"
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / "train_viral_graph.csv"
    val_file = data_dir / "val_viral_graph.csv"
    test_file = data_dir / "test_viral_graph.csv"

    print("=" * 60)
    print("ABLATION: NO GRAPH-TO-PROTEIN CROSS-ATTENTION")
    print("=" * 60, flush=True)

    print("\n[10%] Loading molecular graphs...", flush=True)
    graphs = load_graphs()
    print(f"Loaded {len(graphs):,} graphs", flush=True)

    print("\n[20%] Preparing datasets...", flush=True)
    train_ds = AffinityDataset(str(train_file), mtype="Ki", graphs=graphs)
    val_ds = AffinityDataset(str(val_file), mtype="Ki", graphs=graphs)
    test_ds = AffinityDataset(str(test_file), mtype="Ki", graphs=graphs)

    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise RuntimeError("A dataset is empty.")

    print(f"Train: {len(train_ds):,}")
    print(f"Validation: {len(val_ds):,}")
    print(f"Test: {len(test_ds):,}", flush=True)

    batch_size = 8
    epochs = 10
    learning_rate = 1e-4

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[30%] Device: {device}", flush=True)

    model = AblationNoGraphCross().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    train_df = pd.read_csv(train_file)
    train_df = train_df[
        train_df["mtype"].astype(str).str.lower() == "ki"
    ]
    label_mean = float(train_df["label"].mean())
    print(f"Training label mean: {label_mean:.4f}", flush=True)

    best_path = checkpoint_dir / "viral_ki_no_graph_cross_best.pt"
    best_val_loss = float("inf")

    def run_epoch(loader, training):
        model.train(training)
        total_loss = 0.0
        total_count = 0

        with torch.set_grad_enabled(training):
            for step, batch in enumerate(loader, start=1):
                batch = batch.to(device)
                targets = batch.y.view(-1).float() - label_mean

                if training:
                    optimizer.zero_grad(set_to_none=True)

                predictions = model(batch).view(-1)
                loss = criterion(predictions, targets)

                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )
                    optimizer.step()

                count = targets.numel()
                total_loss += loss.item() * count
                total_count += count

                if training and step % 100 == 0:
                    print(f"  Batch {step}/{len(loader)}", flush=True)

        return total_loss / max(total_count, 1)

    print(f"\n[40%] Training for {epochs} epochs...", flush=True)

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(train_loader, True)
        val_loss = run_epoch(val_loader, False)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_mean": label_mean,
                    "epoch": epoch,
                    "val_mse": val_loss,
                    "model_name": "AblationNoGraphCross",
                    "experiment": "viral_Ki_no_graph_cross",
                    "config": model.cfg,
                },
                best_path,
            )
            print("  Best ablation checkpoint saved.", flush=True)

    print("\n[90%] Evaluating best checkpoint...", flush=True)
    checkpoint = torch.load(
        best_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    actual_all = []
    predicted_all = []

    with torch.no_grad():
        for step, batch in enumerate(test_loader, start=1):
            batch = batch.to(device)
            predictions = model(batch).view(-1) + label_mean
            actual_all.extend(batch.y.view(-1).cpu().numpy().tolist())
            predicted_all.extend(predictions.cpu().numpy().tolist())

            if step % 10 == 0:
                print(
                    f"  Test batch {step}/{len(test_loader)}",
                    flush=True,
                )

    actual = np.asarray(actual_all, dtype=float)
    predicted = np.asarray(predicted_all, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]

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

    print("\n=== NO-GRAPH-CROSS ABLATION RESULTS ===")
    print(f"Valid test samples: {len(actual)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Pearson: {pearson:.4f}")
    print(f"Spearman: {spearman:.4f}")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Best validation MSE: {checkpoint['val_mse']:.4f}")

    output = pd.DataFrame({
        "actual_pAff": actual,
        "predicted_pAff": predicted,
        "absolute_error": np.abs(errors),
    })
    output_path = root / "viral_ki_no_graph_cross_predictions.csv"
    output.to_csv(output_path, index=False)

    print(f"\nPredictions saved to: {output_path}")
    print(f"Checkpoint saved to: {best_path}")
    print("\nAblation training and evaluation finished.", flush=True)


if __name__ == "__main__":
    main()
