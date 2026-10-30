# Drug–Target Binding Affinity Predictor via 3D Graph Transformers

A multimodal deep-learning project for estimating the binding affinity between a small-molecule ligand and a target protein. The project combines **3D molecular graph representations**, **protein sequence representations**, and (in the multimodal experiment) **SMILES sequence representations** to predict drug–target binding affinity.

> **Research project notice:** This repository is an experimental research implementation. Predictions are computational estimates, not experimental measurements or medical advice. Model performance on the reported test split should not be interpreted as proof of generalization to every drug, protein, virus, or biological setting.

---

## Table of contents

- [Overview](#overview)
- [Key features](#key-features)
- [Model architecture](#model-architecture)
- [Repository structure](#repository-structure)
- [Affinity labels](#affinity-labels)
- [Dataset and preprocessing](#dataset-and-preprocessing)
- [Installation](#installation)
- [Running the project](#running-the-project)
- [Training experiments](#training-experiments)
- [Evaluation](#evaluation)
- [Prediction for a new ligand–protein pair](#prediction-for-a-new-ligandprotein-pair)
- [Interactive dashboard](#interactive-dashboard)
- [Reported experimental results](#reported-experimental-results)
- [Screenshots](#screenshots)
- [Outputs and generated files](#outputs-and-generated-files)
- [Reproducibility notes and limitations](#reproducibility-notes-and-limitations)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Acknowledgements](#acknowledgements)

---

## Overview

Drug–target binding affinity describes how strongly a molecule binds to a biological target. Reliable affinity estimation can help prioritize compounds for further investigation, but experimental affinity measurements are expensive and are not always available for every ligand–protein pair.

This project explores an affinity-prediction pipeline built around graph-based molecular learning and protein sequence encoding. A ligand is represented as a molecular graph with atom and bond features and 3D coordinates. A protein is represented as a sequence of amino-acid residues. The encoders learn representations of these inputs, and the model combines the learned information to estimate an affinity value.

The repository contains a baseline graph–protein model, a multimodal model that additionally uses SMILES, and ablation variants used to study the contribution of cross-attention components.

### High-level workflow

1. Clean BindingDB records and retain valid ligand SMILES, protein sequences, and affinity measurements.
2. Convert affinity measurements into a consistent p-affinity target.
3. Create train, validation, and test splits, using scaffold grouping to reduce ligand-scaffold leakage.
4. Generate 3D molecular graphs from SMILES.
5. Load graph and protein information into dataset objects.
6. Train baseline, multimodal, and ablation models.
7. Evaluate predictions and compare models using regression and correlation metrics.
8. Inspect saved results through the Streamlit dashboard.

---

## Key features

- **3D molecular graph representation:** RDKit is used to parse SMILES, generate conformers, and calculate atom and bond features.
- **Graph Transformer:** molecular graph encoding uses graph neural-network operations and spatial information from 3D atom coordinates.
- **Protein sequence encoder:** amino-acid sequences are converted to residue-token representations and processed by a neural encoder.
- **SMILES sequence encoder:** the multimodal variant includes an additional character-level SMILES representation.
- **Cross-attention:** enables information exchange between learned molecular and protein representations.
- **Ablation experiments:** compare the full multimodal model with variants that remove selected cross-attention components.
- **Evaluation and visualization:** saved predictions and model metrics can be inspected in a Streamlit interface.

---

## Model architecture

### Architecture overview

The diagram below summarizes the main data-preparation, modeling, training/evaluation, and analysis components. The file links point to the corresponding source files in this repository.

```mermaid
flowchart TD
    subgraph data["Data preparation"]
        clean["Clean records<br/>clean_bindingdb.py"]
        labels["Prepare affinity labels<br/>prepare_affinity_labels.py"]
        splits["Split datasets<br/>split_bindingdb.py"]
        graphs["Build 3D graphs<br/>build_3d_graphs.py"]
        dataset["Affinity dataset<br/>graph_dataset.py"]
        common["Shared utilities and featurization<br/>common.py"]
    end

    subgraph model["Affinity modeling"]
        graph["3D graph encoder<br/>graph_transformer_3d.py"]
        protein["Protein encoder<br/>protein_encoder.py"]
        smiles["SMILES encoder<br/>smiles_encoder.py"]
        cross["Cross-attention<br/>cross_attention.py"]
        baseline["Baseline affinity model<br/>affinity_model.py"]
        multimodal["Multimodal model<br/>affinity_model_multimodal.py"]
        noGraph["No-graph-cross ablation<br/>affinity_model_no_graph_cross.py"]
        noSmiles["No-SMILES-cross ablation<br/>affinity_model_no_smiles_cross.py"]
    end

    subgraph experiments["Training and evaluation"]
        trainBase["BindingDB training<br/>train_bindingdb.py"]
        trainViral["Viral training<br/>train_viral.py"]
        trainMulti["Multimodal training<br/>train_viral_multimodal.py"]
        trainNoGraph["No-graph-cross training<br/>train_viral_no_graph_cross.py"]
        trainNoSmiles["No-SMILES-cross training<br/>train_viral_no_smiles_cross.py"]
        evalBase["BindingDB evaluation<br/>evaluate_bindingdb.py"]
        evalViral["Viral evaluation<br/>evaluate_viral.py"]
        evalMulti["Multimodal evaluation<br/>evaluate_multimodal.py"]
    end

    subgraph interface["Prediction and analysis"]
        predict["Affinity prediction<br/>predict_affinity.py"]
        app["Streamlit dashboard<br/>app.py"]
        compare["Model comparison<br/>compare_all_models.py"]
        target["Target analysis<br/>analyze_targets.py"]
    end

    bindingdb[("BindingDB records")]

    bindingdb --> clean
    clean --> labels
    labels --> splits
    splits --> graphs
    clean --> common
    labels --> common
    splits --> common
    graphs --> common
    dataset --> common
    graphs --> dataset
    dataset --> graph

    baseline --> graph
    baseline --> protein
    baseline --> cross
    multimodal --> graph
    multimodal --> protein
    multimodal --> smiles
    multimodal --> cross
    noGraph --> graph
    noGraph --> protein
    noGraph --> smiles
    noGraph --> cross
    noSmiles --> graph
    noSmiles --> protein
    noSmiles --> smiles
    noSmiles --> cross

    trainBase --> dataset
    trainBase --> baseline
    trainViral --> dataset
    trainViral --> baseline
    trainMulti --> dataset
    trainMulti --> multimodal
    trainNoGraph --> dataset
    trainNoGraph --> noGraph
    trainNoSmiles --> dataset
    trainNoSmiles --> noSmiles

    evalBase --> dataset
    evalBase --> baseline
    evalViral --> dataset
    evalViral --> baseline
    evalMulti --> dataset
    evalMulti --> multimodal

    predict --> common
    predict --> evalBase
    app --> compare
    compare --> target

    classDef dataStyle fill:#dbeafe,stroke:#2563eb,color:#172554
    classDef modelStyle fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef expStyle fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef interfaceStyle fill:#ffe4e6,stroke:#e11d48,color:#881337
    class clean,labels,splits,graphs,dataset,common dataStyle
    class graph,protein,smiles,cross,baseline,multimodal,noGraph,noSmiles modelStyle
    class trainBase,trainViral,trainMulti,trainNoGraph,trainNoSmiles,evalBase,evalViral,evalMulti expStyle
    class predict,app,compare,target interfaceStyle

    click clean "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/clean_bindingdb.py"
    click labels "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/prepare_affinity_labels.py"
    click splits "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/split_bindingdb.py"
    click graphs "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/build_3d_graphs.py"
    click dataset "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/graph_dataset.py"
    click common "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/common.py"
    click graph "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/graph_transformer_3d.py"
    click protein "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/protein_encoder.py"
    click smiles "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/smiles_encoder.py"
    click cross "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/cross_attention.py"
    click baseline "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/affinity_model.py"
    click multimodal "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/affinity_model_multimodal.py"
    click noGraph "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/affinity_model_no_graph_cross.py"
    click noSmiles "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/affinity_model_no_smiles_cross.py"
    click trainBase "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/train_bindingdb.py"
    click trainViral "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/train_viral.py"
    click trainMulti "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/train_viral_multimodal.py"
    click trainNoGraph "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/train_viral_no_graph_cross.py"
    click trainNoSmiles "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/train_viral_no_smiles_cross.py"
    click evalBase "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/evaluate_bindingdb.py"
    click evalViral "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/evaluate_viral.py"
    click evalMulti "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/evaluate_multimodal.py"
    click predict "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/predict_affinity.py"
    click app "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/app.py"
    click compare "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/compare_all_models.py"
    click target "https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers/blob/main/src/analyze_targets.py"
```

### 3D molecular graph encoder

Each ligand begins as a SMILES string. RDKit parses the molecule, adds hydrogens to generate a 3D conformer, performs an MMFF optimization attempt, and then retains heavy atoms for graph construction.

The shared featurization code defines:

- **23 atom features**, including atom identity, degree, formal charge, aromaticity, hybridization, and hydrogen count.
- **6 bond features**, including bond type, conjugation, and ring membership.
- **3D atom coordinates**, used to represent spatial relationships.
- A maximum of **80 heavy atoms** per molecule in the shared graph-building utility.

The graph encoder uses graph-based message passing and Transformer-style spatial attention. Pairwise atom distances can be expanded using Gaussian radial basis functions and incorporated as attention biases, allowing the attention mechanism to use 3D geometry as well as learned feature similarity.

For atoms *i* and *j*, the Euclidean distance is:

\[
d_{ij} = \|\mathbf{r}_i-\mathbf{r}_j\|_2
\]

A Gaussian radial basis function can encode this distance:

\[
\phi_k(d_{ij}) = \exp\left[-\gamma(d_{ij}-\mu_k)^2\right]
\]

The distance features contribute to a head-specific attention bias:

\[
B_h[i,j] = \mathbf{w}_h^\top\boldsymbol{\phi}(d_{ij})+b_h
\]

Spatially biased self-attention is then:

\[
\operatorname{Attention}_h =
\operatorname{softmax}\left(
\frac{Q_hK_h^\top}{\sqrt{d_h}}+B_h
\right)V_h
\]

### Protein encoder

Protein sequences are represented using amino-acid residue tokens. The shared utility encodes the 20 standard amino acids and pads or truncates sequences to a maximum length of **1,000 residues**. The protein encoder processes these residue representations to produce a learned protein representation for affinity prediction.

### SMILES encoder and multimodal fusion

The multimodal model adds a character-level SMILES encoder alongside the 3D molecular graph encoder and protein encoder. Cross-attention layers allow learned representations from different input modalities to interact before the regression head predicts affinity.

The repository also contains ablation models that remove selected cross-attention pathways. These variants help assess whether the additional modality interactions improve the measured results.

---

## Repository structure

The repository's tracked files include `src/`, `results/`, `requirements.txt`, model-comparison output, and saved viral test-prediction CSV files. The data, graph, checkpoint, and log directories are created or populated during preprocessing and experiments; large raw datasets and generated artifacts may not be included in a fresh clone.

```text
drug-target-binding-affinity-predictor-via-graph-transformers/
│
├── src/
│   ├── clean_bindingdb.py
│   ├── prepare_affinity_labels.py
│   ├── split_bindingdb.py
│   ├── build_3d_graphs.py
│   ├── graph_dataset.py
│   ├── common.py
│   │
│   ├── graph_transformer_3d.py
│   ├── protein_encoder.py
│   ├── smiles_encoder.py
│   ├── cross_attention.py
│   ├── affinity_model.py
│   ├── affinity_model_multimodal.py
│   ├── affinity_model_no_graph_cross.py
│   ├── affinity_model_no_smiles_cross.py
│   │
│   ├── train_bindingdb.py
│   ├── train_viral.py
│   ├── train_viral_multimodal.py
│   ├── train_viral_no_graph_cross.py
│   ├── train_viral_no_smiles_cross.py
│   │
│   ├── evaluate_bindingdb.py
│   ├── evaluate_viral.py
│   ├── evaluate_multimodal.py
│   ├── predict_affinity.py
│   ├── compare_all_models.py
│   ├── analyze_targets.py
│   └── app.py
│
├── data/
│   ├── raw/                 # Place the BindingDB TSV file here
│   ├── processed/           # Cleaned, labeled, and split CSV files
│   └── graphs/              # Generated PyTorch Geometric graph shards
│
├── checkpoints/             # Viral experiment checkpoints
├── models/                  # BindingDB model checkpoints
├── results/
│   ├── metrics/
│   └── plots/
├── logs/
│
├── all_model_comparison.csv
├── viral_ki_test_predictions.csv
├── viral_ki_multimodal_test_predictions.csv
├── viral_ki_no_graph_cross_predictions.csv
├── viral_ki_no_smiles_cross_predictions.csv
├── requirements.txt
└── README.md
```

> The tree above documents the project layout and expected generated directories. A fresh clone may not contain every generated directory or data file. The `common.py` utility creates several working directories when the preprocessing/model modules are imported.

---

## Affinity labels

The project converts affinity measurements recorded in nanomolar (nM) units to a p-affinity scale:

\[
\boxed{\mathrm{pAff}=9-\log_{10}(K\ \mathrm{in\ nM})}
\]

Here, \(K\) denotes the affinity measurement for the relevant measurement type (for example, Ki, Kd, IC50, or EC50). Higher p-affinity values correspond to lower nanomolar affinity values.

The label-preparation script:

- Parses numeric affinity values and optional inequality qualifiers.
- Filters values outside the configured range of 0.01 to \(10^7\) nM.
- Keeps measurement types separate rather than mixing Ki, Kd, IC50, and EC50.
- Aggregates repeated measurements by ligand, protein sequence, measurement type, and qualifier using the median.
- Drops censored (`<` or `>`) measurements by default. The script supports `--keep-censored` to retain them.

For the viral Ki experiments, the training scripts filter the data to `mtype = Ki`.

---

## Dataset and preprocessing

### Data source

The preprocessing scripts expect a BindingDB tab-separated (`.tsv`) file under:

```text
data/raw/
```

The shared utility searches this directory for TSV files and selects the last file in sorted filename order. Make sure the intended BindingDB file is the one selected.

Obtain the dataset from [BindingDB](https://www.bindingdb.org/) and follow its data-use and citation requirements. The full raw dataset is not assumed to be included in this GitHub repository.

### Preprocessing stages

The source code implements the following stages:

| Stage | Script | Purpose |
|---|---|---|
| Cleaning | `src/clean_bindingdb.py` | Retains valid SMILES, protein sequences, and records with at least one affinity measurement |
| Label preparation | `src/prepare_affinity_labels.py` | Converts affinity values into p-affinity labels and aggregates repeated measurements |
| Dataset splitting | `src/split_bindingdb.py` | Creates scaffold-grouped train, validation, and test CSV files and checks for scaffold/pair leakage |
| 3D graph generation | `src/build_3d_graphs.py` | Generates and stores molecular graph objects in PyTorch `.pt` shards |
| Dataset loading | `src/graph_dataset.py` | Loads graph shards and constructs model-ready affinity examples |

### Scaffold split

The splitting script groups ligands by Bemis–Murcko scaffold and assigns scaffold groups to partitions. Its default target proportions are approximately:

- 80% training
- 10% validation
- 10% test

The script includes checks to ensure that scaffolds do not overlap between train, validation, and test partitions, and checks ligand–protein pair overlap between train and test. The exact resulting counts can differ from the target proportions because whole scaffold groups are assigned together.

### Viral subset note

The viral experiment scripts expect the following graph-ready CSV files:

```text
data/processed/
├── train_viral_graph.csv
├── val_viral_graph.csv
└── test_viral_graph.csv
```

These files must exist before running the viral training scripts. The repository's viral training scripts load these exact paths. The generic cleaning, label preparation, and scaffold-splitting scripts described above should not be assumed to create these three viral graph-ready files automatically. Prepare the viral subset and its graph-ready split files using the project's intended experiment data-preparation workflow before launching viral training.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers.git
cd drug-target-binding-affinity-predictor-via-graph-transformers
```

### 2. Create a Python environment

Python 3.10 or 3.11 is a reasonable starting point for this dependency stack. Use a clean environment to avoid conflicts between PyTorch, PyTorch Geometric, and RDKit.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

The repository's `requirements.txt` lists:

- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- rdkit
- torch
- torch_geometric
- streamlit

Install the listed packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**PyTorch/PyTorch Geometric compatibility:** if the default installation does not provide a compatible PyTorch Geometric environment, install the appropriate PyTorch build for your operating system and CPU/CUDA setup first, then follow the official [PyTorch Geometric installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

Check the environment:

```bash
python -c "import torch, torch_geometric, rdkit, pandas, streamlit; print('torch:', torch.__version__); print('PyG:', torch_geometric.__version__); print('CUDA available:', torch.cuda.is_available()); print('Environment OK')"
```

---

## Running the project

Run commands from the repository root unless a command explicitly changes directory.

### Step 1 — Add the raw BindingDB file

Create the raw-data directory and place the downloaded TSV file inside it:

```bash
mkdir -p data/raw
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force data/raw
```

Expected location:

```text
data/raw/your_bindingdb_export.tsv
```

The preprocessing utility expects a `.tsv` file. It does not download BindingDB automatically.

### Step 2 — Clean BindingDB records

```bash
python src/clean_bindingdb.py
```

Optional: process only an initial number of raw rows for a quick pipeline test:

```bash
python src/clean_bindingdb.py 100000
```

The cleaning script writes:

```text
data/processed/bindingdb_clean.csv
logs/clean_report.csv
```

The optional row limit is useful for a smoke test; it is not a substitute for running the complete dataset when reproducing the reported experiment.

### Step 3 — Prepare affinity labels

```bash
python src/prepare_affinity_labels.py
```

To retain censored measurements:

```bash
python src/prepare_affinity_labels.py --keep-censored
```

Default behavior drops measurements with inequality qualifiers. The output is:

```text
data/processed/bindingdb_labels.csv
```

### Step 4 — Create scaffold-based splits

```bash
python src/split_bindingdb.py --seed 42
```

For a smaller development subset:

```bash
python src/split_bindingdb.py --subset 50000 --seed 42
```

The generic split script writes:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
logs/split_info.json
```

It also runs scaffold and ligand–protein pair leakage checks.

### Step 5 — Build 3D molecular graphs

```bash
python src/build_3d_graphs.py --mtype Ki --workers 6 --batch-size 500
```

Optional arguments:

| Argument | Meaning |
|---|---|
| `--mtype` | Affinity measurement type; defaults to `Ki` |
| `--workers` | Number of worker processes; defaults to `6` |
| `--batch-size` | Number of successfully generated graphs per saved shard; defaults to `500` |
| `--limit` | Optional limit on the number of unique SMILES processed |

For a small test:

```bash
python src/build_3d_graphs.py --mtype Ki --workers 2 --batch-size 50 --limit 100
```

Generated graph shards are stored under:

```text
data/graphs/
```

The graph builder can skip SMILES already present in existing graph shards. Molecules whose conformers cannot be generated are recorded in `logs/failed_conformers.txt`.

### Step 6 — Prepare the viral graph-ready split files

Before running the viral experiments, verify that these files are present:

```text
data/processed/train_viral_graph.csv
data/processed/val_viral_graph.csv
data/processed/test_viral_graph.csv
```

The viral training and evaluation scripts explicitly load these files and filter examples to Ki measurements. The generic `split_bindingdb.py` command produces `train.csv`, `val.csv`, and `test.csv`; it does not, by itself, produce the three viral graph-ready filenames above. Ensure that the viral subset selection and graph-ready CSV preparation step has been completed according to the experiment setup before continuing.

### Step 7 — Train a model

Choose one of the training experiments in the [Training experiments](#training-experiments) section. For example:

```bash
python src/train_viral.py
```

### Step 8 — Evaluate and inspect predictions

For the baseline viral model:

```bash
python src/evaluate_viral.py
```

The evaluation script writes:

```text
viral_ki_test_predictions.csv
```

### Step 9 — Launch the dashboard

```bash
streamlit run src/app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Open that address in your browser.

---

## Training experiments

The repository contains multiple training scripts for model comparison.

| Experiment | Script | Model/checkpoint purpose |
|---|---|---|
| BindingDB baseline | `src/train_bindingdb.py` | Trains the baseline graph–protein affinity model for a selected measurement type |
| Viral Ki baseline | `src/train_viral.py` | Trains the baseline model on viral Ki graph-ready splits |
| Viral multimodal | `src/train_viral_multimodal.py` | Trains the model using graph, protein, and SMILES representations |
| No graph cross-attention | `src/train_viral_no_graph_cross.py` | Trains the corresponding graph-cross-attention ablation |
| No SMILES cross-attention | `src/train_viral_no_smiles_cross.py` | Trains the corresponding SMILES-cross-attention ablation |

### BindingDB baseline training

The BindingDB training script accepts configurable arguments:

```bash
python src/train_bindingdb.py --mtype Ki --epochs 20 --batch 16 --lr 0.0003 --tag model_Ki
```

Useful arguments:

| Argument | Default | Description |
|---|---:|---|
| `--mtype` | `Ki` | Measurement type |
| `--epochs` | `20` | Maximum training epochs |
| `--batch` | `16` | Batch size |
| `--lr` | `3e-4` | Learning rate |
| `--max-train` | None | Optional cap on training examples |
| `--patience` | `5` | Early-stopping patience |
| `--d` | `128` | Model dimension |
| `--tag` | `model_<mtype>` | Checkpoint name |
| `--seed` | `42` | Random seed |

The script saves the checkpoint under `models/` and the training history under `results/metrics/`.

### Viral baseline

```bash
python src/train_viral.py
```

The current script uses a batch size of 8, 10 epochs, and a learning rate of `1e-4`. It selects the best checkpoint using validation MSE and writes:

```text
checkpoints/viral_ki_best.pt
```

### Viral multimodal model

```bash
python src/train_viral_multimodal.py
```

The multimodal training script writes its best checkpoint to:

```text
checkpoints/viral_ki_multimodal_best.pt
```

### Ablation: no graph cross-attention

```bash
python src/train_viral_no_graph_cross.py
```

This experiment writes:

```text
checkpoints/viral_ki_no_graph_cross_best.pt
```

### Ablation: no SMILES cross-attention

```bash
python src/train_viral_no_smiles_cross.py
```

This experiment writes:

```text
checkpoints/viral_ki_no_smiles_cross_best.pt
viral_ki_no_smiles_cross_predictions.csv
```

> Run each viral experiment only after the viral graph-ready CSV files and molecular graph shards are available. Training from scratch can take significant time and memory depending on the dataset and hardware.

---

## Evaluation

### BindingDB model evaluation

For a BindingDB checkpoint created with `--tag model_Ki`:

```bash
python src/evaluate_bindingdb.py --tag model_Ki --split test
```

Optional batch size:

```bash
python src/evaluate_bindingdb.py --tag model_Ki --split test --batch 32
```

The evaluation script reports metrics and saves prediction and metric files under `results/metrics/`.

### Viral baseline evaluation

```bash
python src/evaluate_viral.py
```

This script loads `checkpoints/viral_ki_best.pt`, evaluates the test split, and reports:

- MAE (mean absolute error)
- RMSE (root mean squared error)
- Pearson correlation
- Spearman correlation
- Mean-baseline MAE and RMSE

It also writes `viral_ki_test_predictions.csv` to the repository root.

### Metrics explained

| Metric | What it measures | Preferred direction |
|---|---|---|
| MAE | Average absolute difference between predicted and actual pAff | Lower |
| RMSE | Square root of average squared prediction error; penalizes larger errors more strongly | Lower |
| Pearson correlation | Linear association between predictions and actual values | Higher |
| Spearman correlation | Rank-order association between predictions and actual values | Higher |

---

## Prediction for a new ligand–protein pair

The repository includes `src/predict_affinity.py` for command-line inference using a compatible **BindingDB baseline checkpoint**.

Example:

```bash
python src/predict_affinity.py \
  --tag model_Ki \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --seq "MKT..."
```

Replace the example sequence with a valid amino-acid sequence. The `--tag` value must match a checkpoint in `models/`, including its measurement type and model configuration.

The prediction function returns:

- `mtype`: affinity measurement type stored in the checkpoint.
- `pAffinity`: predicted p-affinity.
- `nM`: affinity converted back to nanomolar units.

The conversion used is:

\[
K_{\mathrm{nM}}=10^{(9-\mathrm{pAff})}
\]

**Important:** the prediction script loads the baseline model through `evaluate_bindingdb.py`. It should not be assumed to load the viral multimodal or ablation checkpoints. Also, the current Streamlit dashboard is for inspecting saved experiment outputs; it does not provide a form for submitting a new molecule and protein sequence.

---

## Interactive dashboard

Launch the application from the repository root:

```bash
streamlit run src/app.py
```

The dashboard contains the following pages:

### Home

- Project summary and quick experiment statistics.
- Model performance chart.
- Best MAE and RMSE summaries.
- All-model comparison table.

### Model Comparison

- Compare the four saved model variants.
- Select MAE, RMSE, Pearson, or Spearman.
- View a chart and comparison table.
- Download the comparison CSV.

### Predictions

- Select a model's saved test-prediction CSV.
- View row count, MAE, and RMSE.
- Inspect actual versus predicted pAff in a scatter plot.
- Review prediction rows and download the displayed predictions.

### Dataset Overview

- Detect available processed CSV datasets.
- Display row and column counts.
- Preview target-summary data when available.

### Project Files

- Inspect available checkpoint, chart, log, and processed-data files.

The dashboard reads saved result files from the repository. If a result CSV or processed dataset is missing, the corresponding page may show a warning or an empty view.

---

## Reported experimental results

The repository includes `all_model_comparison.csv` and prediction CSV files for four viral Ki model variants. The dashboard's fallback comparison values are:

| Model | Test samples | MAE | RMSE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| Baseline | 547 | 0.7879 | 1.0298 | 0.7947 | 0.7797 |
| Full multimodal | 547 | 0.7571 | 1.0262 | 0.7923 | 0.7792 |
| No graph cross-attention (C) | 547 | 0.7583 | 0.9978 | 0.7997 | 0.7882 |
| No SMILES cross-attention (D) | 547 | 0.8382 | 1.0753 | 0.7809 | 0.7721 |

**Interpretation:** in these reported results, the full multimodal model has the lowest MAE, while the no-graph-cross-attention variant has the lowest RMSE and the highest Pearson and Spearman correlations. The no-SMILES-cross-attention variant has the highest error values among the four. These are results from the reported test split and experiments; they should not be generalized beyond the evaluated setting without additional validation.

The dashboard prefers the repository's comparison CSV when its columns are compatible and otherwise uses the fallback values shown above. If you rerun experiments, verify that the CSV corresponds to the checkpoints and prediction files you intend to report.

---

## Screenshots

Add screenshots of your own application and experiment outputs to the `docs/images/` directory. Then replace the placeholder paths below with the filenames you save.

Suggested folder:

```text
docs/
└── images/
    ├── dashboard-home.png
    ├── model-comparison.png
    ├── predictions-page.png
    ├── dataset-overview.png
    ├── project-files.png
    └── terminal-training.png
```

### 1. Dashboard — Home

<!-- Replace this path after adding your screenshot. -->
![Dashboard home](docs/images/dashboard-home.png)

### 2. Model comparison

<!-- Replace this path after adding your screenshot. -->
![Model comparison](docs/images/model-comparison.png)

### 3. Prediction analysis

<!-- Replace this path after adding your screenshot. -->
![Prediction analysis](docs/images/predictions-page.png)

### 4. Dataset overview

<!-- Replace this path after adding your screenshot. -->
![Dataset overview](docs/images/dataset-overview.png)

### 5. Project files

<!-- Replace this path after adding your screenshot. -->
![Project files](docs/images/project-files.png)

### 6. Training output

<!-- Replace this path after adding your screenshot. -->
![Training output](docs/images/terminal-training.png)

**How to add screenshots**

1. Run the dashboard using `streamlit run src/app.py`.
2. Open each dashboard page and capture a screenshot.
3. Create `docs/images/` in the repository.
4. Save screenshots using the filenames listed above.
5. Commit and push the images along with the README.

For a screenshot on most systems:
- Windows: `Win + Shift + S`
- macOS: `Shift + Command + 4`
- Linux: use the desktop screenshot utility.

---

## Outputs and generated files

| Path | Description |
|---|---|
| `data/processed/bindingdb_clean.csv` | Cleaned BindingDB records |
| `data/processed/bindingdb_labels.csv` | Long-format affinity labels |
| `data/processed/train.csv` | Training split |
| `data/processed/val.csv` | Validation split |
| `data/processed/test.csv` | Test split |
| `data/graphs/graphs_*.pt` | Serialized molecular graph shards |
| `models/*.pt` | BindingDB model checkpoints |
| `checkpoints/*.pt` | Viral experiment checkpoints |
| `results/metrics/` | Training histories and evaluation metrics/predictions |
| `results/plots/` | Experiment plots, when generated |
| `logs/` | Cleaning, splitting, and conformer-generation logs |
| `all_model_comparison.csv` | Comparison of model variants |
| `viral_ki_*_predictions.csv` | Saved viral Ki test predictions |

---

## Reproducibility notes and limitations

- The BindingDB source file is not included by default; obtain it separately and place it under `data/raw/`.
- The graph-building stage can be computationally expensive because it generates 3D conformers for unique molecules.
- The graph featurization utility rejects invalid SMILES, molecules with fewer than two atoms, molecules with more than 80 heavy atoms, and molecules for which conformer generation fails.
- Protein sequences are encoded up to 1,000 residues by the shared utility.
- The viral training scripts expect pre-created `train_viral_graph.csv`, `val_viral_graph.csv`, and `test_viral_graph.csv` files.
- Random seeds are set in the training scripts, but exact reproducibility can still depend on hardware, package versions, and CUDA operations.
- Reported model metrics are based on a particular test split and run. Multiple seeds, external test sets, and target-level generalization studies are needed for stronger conclusions.
- Affinity prediction is not equivalent to experimentally measuring binding, efficacy, toxicity, or clinical suitability.

---

## Troubleshooting

### `FileNotFoundError: No .tsv in data/raw`

Place a BindingDB `.tsv` file in `data/raw/`. Check that it has the `.tsv` extension.

### `Required dataset not found: train_viral_graph.csv`

The viral training scripts require graph-ready viral train, validation, and test CSVs under `data/processed/`. The generic scaffold split produces `train.csv`, `val.csv`, and `test.csv`; it does not automatically create those viral-specific filenames.

### `No molecular graphs found` or empty dataset

Confirm that graph shards were generated under `data/graphs/`, that the graph files match the SMILES in the selected split, and that the requested measurement type (for example, `Ki`) is present.

### RDKit conformer generation failures

Some SMILES cannot be embedded into a valid 3D conformer. Review `logs/failed_conformers.txt`. The graph builder continues processing other molecules when individual conformers fail.

### PyTorch Geometric installation errors

Check the installed PyTorch version and install a compatible PyTorch Geometric build using its official installation guide. Also verify whether the environment is intended to use CPU or CUDA.

### Streamlit dashboard shows missing files

The dashboard reads prediction CSVs and comparison results from the repository root. Confirm that the expected CSV files exist and that you launched Streamlit from the repository root.

---

## References

- [Project repository](https://github.com/parri-abinaya/drug-target-binding-affinity-predictor-via-graph-transformers)
- [BindingDB](https://www.bindingdb.org/)
- [RDKit documentation](https://www.rdkit.org/docs/)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/)
- [Streamlit documentation](https://docs.streamlit.io/)

## Acknowledgements

This project uses BindingDB affinity data and open-source scientific Python tools, including RDKit, PyTorch, PyTorch Geometric, pandas, NumPy, SciPy, scikit-learn, Matplotlib, and Streamlit. Please cite the relevant dataset and software references when using this work in academic or research settings.
