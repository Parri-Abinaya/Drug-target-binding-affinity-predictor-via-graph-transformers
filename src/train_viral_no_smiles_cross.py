import os
import random
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from graph_dataset import load_graphs, AffinityDataset
from affinity_model_no_smiles_cross import AblationNoSmilesCross

SEED = 42
EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

graphs = load_graphs()
base = "..\\data\\processed"

train_csv = os.path.join(base, "train_viral_graph.csv")
val_csv = os.path.join(base, "val_viral_graph.csv")
test_csv = os.path.join(base, "test_viral_graph.csv")

train_ds = AffinityDataset(train_csv, mtype="Ki", graphs=graphs)
val_ds = AffinityDataset(val_csv, mtype="Ki", graphs=graphs)
test_ds = AffinityDataset(test_csv, mtype="Ki", graphs=graphs)

print("Dataset sizes:", len(train_ds), len(val_ds), len(test_ds))

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False
)

# Center labels using the training-set mean only.
train_labels = np.array([
    float(train_ds[i].y.item()) for i in range(len(train_ds))
])
label_mean = float(train_labels.mean())
print("Training label mean:", label_mean)

model = AblationNoSmilesCross().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss()

os.makedirs("..\\checkpoints", exist_ok=True)
checkpoint_path = os.path.join(
    "..", "checkpoints", "viral_ki_no_smiles_cross_best.pt"
)

best_val = float("inf")
best_epoch = 0

def run_epoch(loader, training):
    model.train(training)
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        batch = batch.to(device)
        labels = batch.y.view(-1).float() - label_mean

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predictions = model(batch)
            loss = criterion(predictions.view(-1), labels)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        count = labels.numel()
        total_loss += loss.item() * count
        total_count += count

    return total_loss / max(total_count, 1)

for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_loader, True)
    val_loss = run_epoch(val_loader, False)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}"
    )

    if val_loss < best_val:
        best_val = val_loss
        best_epoch = epoch
        torch.save({
            "model_state_dict": model.state_dict(),
            "label_mean": label_mean,
            "best_val_mse": best_val,
            "best_epoch": best_epoch,
            "seed": SEED,
        }, checkpoint_path)
        print("  Saved best checkpoint:", checkpoint_path)

# Load the best validation checkpoint before generating test predictions.
checkpoint = torch.load(
    checkpoint_path, map_location=device, weights_only=False
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_true = []
all_pred = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch).view(-1) + label_mean
        actual = batch.y.view(-1).float()

        all_true.extend(actual.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())

out_path = "..\\viral_ki_no_smiles_cross_predictions.csv"
pd.DataFrame({
    "actual_pAff": all_true,
    "predicted_pAff": all_pred,
}).to_csv(out_path, index=False)

print("\nTraining complete.")
print("Best epoch:", best_epoch)
print("Best validation MSE:", round(best_val, 4))
print("Test predictions saved to:", os.path.abspath(out_path))
print("Test samples:", len(all_true))
