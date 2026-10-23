"""Train the multimodal viral drug-target affinity model."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from graph_dataset import AffinityDataset, load_graphs
from affinity_model_multimodal import MultimodalAffinityModel


def main():
    print("=" * 60)
    print("MULTIMODAL VIRAL DRUG-TARGET AFFINITY TRAINING")
    print("=" * 60, flush=True)

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

    for file in (train_file, val_file, test_file):
        if not file.exists():
            raise FileNotFoundError(f"Dataset not found: {file}")

    print("\n[10%] Loading molecular graphs...", flush=True)
    graphs = load_graphs()
    print(f"Loaded {len(graphs):,} molecular graphs", flush=True)

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

    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise RuntimeError("A Ki dataset is empty.")

    print(f"Training:   {len(train_ds):,}")
    print(f"Validation: {len(val_ds):,}")
    print(f"Test:       {len(test_ds):,}", flush=True)

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

    model = MultimodalAffinityModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate
    )
    criterion = torch.nn.MSELoss()

    train_labels = pd.read_csv(train_file)
    train_labels = train_labels[
        train_labels["mtype"].astype(str).str.lower() == "ki"
    ]
    label_mean = float(train_labels["label"].mean())

    print(f"Training-label mean: {label_mean:.4f}", flush=True)

    best_path = checkpoint_dir / "viral_ki_multimodal_best.pt"
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
                    print(
                        f"  Batch {step}/{len(loader)}",
                        flush=True,
                    )

        return total_loss / max(total_count, 1)

    print(f"\n[40%] Training for {epochs} epochs...", flush=True)

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
                    "model_name": "MultimodalAffinityModel",
                    "experiment": "viral_Ki_multimodal",
                    "config": model.cfg,
                },
                best_path,
            )
            print("  Best multimodal checkpoint saved.", flush=True)

        print(
            f"Training progress: {40 + int(50 * epoch / epochs)}%",
            flush=True,
        )

    print("\n[90%] Evaluating best checkpoint on test set...", flush=True)
    checkpoint = torch.load(
        best_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # The test set is used only after model selection is complete.
    test_mse = run_epoch(test_loader, False)
    print(f"Test MSE:  {test_mse:.4f}")
    print(f"Test RMSE: {test_mse ** 0.5:.4f}")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Best validation MSE: {checkpoint['val_mse']:.4f}")
    print(f"Checkpoint: {best_path}")
    print("\n[100%] Multimodal training and evaluation finished.")


if __name__ == "__main__":
    main()
