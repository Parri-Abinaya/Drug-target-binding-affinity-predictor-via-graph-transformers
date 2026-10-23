"""Train a viral-only drug-target affinity prediction model."""

import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from graph_dataset import AffinityDataset, load_graphs
from affinity_model import AffinityModel


def main():
    print("=" * 60)
    print("VIRAL DRUG-TARGET AFFINITY TRAINING")
    print("=" * 60)

    # Reproducibility
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

    # Initial experiment: viral Ki affinity data
    train_file = data_dir / "train_viral_graph.csv"
    val_file = data_dir / "val_viral_graph.csv"
    test_file = data_dir / "test_viral_graph.csv"

    for file in (train_file, val_file, test_file):
        if not file.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {file}"
            )

    print("\n[10%] Loading molecular graphs...", flush=True)
    graphs = load_graphs()
    print(f"Available molecular graphs: {len(graphs):,}")

    print("\n[20%] Preparing Ki datasets...", flush=True)

    train_ds = AffinityDataset(
        str(train_file), mtype="Ki", graphs=graphs
    )
    val_ds = AffinityDataset(
        str(val_file), mtype="Ki", graphs=graphs
    )
    test_ds = AffinityDataset(
        str(test_file), mtype="Ki", graphs=graphs
    )

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(
            "At least one dataset is empty. Check the Ki filter "
            "and molecular graph coverage."
        )

    print(f"Training examples:   {len(train_ds):,}")
    print(f"Validation examples: {len(val_ds):,}")
    print(f"Test examples:       {len(test_ds):,}")

    # Keep the initial experiment manageable.
    batch_size = 8
    epochs = 10
    learning_rate = 1e-4

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"\n[30%] Device: {device}", flush=True)

    print("Initializing model...", flush=True)
    model = AffinityModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate
    )
    criterion = torch.nn.MSELoss()

    # Center labels using training data only.
    train_labels = pd.read_csv(train_file)
    train_labels = train_labels[
        train_labels["mtype"].astype(str).str.lower() == "ki"
    ]
    label_mean = float(train_labels["label"].mean())

    print(f"Training-label mean: {label_mean:.4f}")
    print(f"\n[40%] Starting training for {epochs} epochs...")
    print("Note: this is a baseline using the current graph-protein model.")

    best_val_loss = float("inf")
    best_path = checkpoint_dir / "viral_ki_best.pt"

    def run_epoch(loader, training):
        model.train(training)
        total_loss = 0.0
        total_count = 0

        with torch.set_grad_enabled(training):
            for batch in loader:
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

        return total_loss / max(total_count, 1)

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(train_loader, True)
        val_loss = run_epoch(val_loader, False)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Val MSE: {val_loss:.4f}",
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
                    "model_name": "AffinityModel",
                    "experiment": "viral_Ki_baseline",
                },
                best_path,
            )
            print("  Best checkpoint saved.", flush=True)

        progress = 40 + int(50 * epoch / epochs)
        print(f"Training progress: {progress}%", flush=True)

    print("\n[90%] Evaluating the best checkpoint...", flush=True)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_mse = run_epoch(test_loader, False)
    test_rmse = test_mse ** 0.5

    print(f"Test MSE:  {test_mse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Best checkpoint: {best_path}")
    print("\n[100%] Training and evaluation finished.")


if __name__ == "__main__":
    main()


