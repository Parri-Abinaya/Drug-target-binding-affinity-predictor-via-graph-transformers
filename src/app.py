from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# This file belongs in: bindingdb_project/src/app.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

COMPARISON_FILE = PROJECT_ROOT / "all_model_comparison.csv"
PREDICTION_FILES = {
    "Baseline": PROJECT_ROOT / "viral_ki_test_predictions.csv",
    "Full multimodal": PROJECT_ROOT / "viral_ki_multimodal_test_predictions.csv",
    "No graph cross-attention (C)": PROJECT_ROOT / "viral_ki_no_graph_cross_predictions.csv",
    "No SMILES cross-attention (D)": PROJECT_ROOT / "viral_ki_no_smiles_cross_predictions.csv",
}

st.set_page_config(
    page_title="Drug Affinity Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light, presentation-friendly styling
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: #f5f8ff; border: 1px solid #dce6fb;
        padding: 14px 16px; border-radius: 12px;
    }
    div[data-testid="stSidebar"] {background: #101d38;}
    div[data-testid="stSidebar"] * {color: #f3f6ff;}
    .small-note {color: #64748b; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

def read_csv(path):
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
    return pd.DataFrame()


def normalize_metrics(frame):
    """Normalize common metric column spellings to consistent names."""
    if frame.empty:
        return frame
    rename = {}
    for col in frame.columns:
        key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        if key in {"model_name", "model"}:
            rename[col] = "Model"
        elif key in {"n", "count", "samples", "num_samples"}:
            rename[col] = "N"
        elif key in {"mae", "mean_absolute_error"}:
            rename[col] = "MAE"
        elif key in {"rmse", "root_mean_squared_error"}:
            rename[col] = "RMSE"
        elif key in {"pearson", "pearson_r", "pearson_correlation"}:
            rename[col] = "Pearson"
        elif key in {"spearman", "spearman_r", "spearman_correlation"}:
            rename[col] = "Spearman"
    return frame.rename(columns=rename)


def load_comparison():
    frame = normalize_metrics(read_csv(COMPARISON_FILE))
    required = {"Model", "MAE", "RMSE", "Pearson", "Spearman"}
    if not frame.empty and required.issubset(frame.columns):
        return frame
    # Fallback to the documented results if the CSV is missing or uses an
    # unexpected schema. The source CSV is still preferred when compatible.
    return pd.DataFrame([
        {"Model": "Baseline", "N": 547, "MAE": 0.7879, "RMSE": 1.0298, "Pearson": 0.7947, "Spearman": 0.7797},
        {"Model": "Full multimodal", "N": 547, "MAE": 0.7571, "RMSE": 1.0262, "Pearson": 0.7923, "Spearman": 0.7792},
        {"Model": "No graph cross-attention (C)", "N": 547, "MAE": 0.7583, "RMSE": 0.9978, "Pearson": 0.7997, "Spearman": 0.7882},
        {"Model": "No SMILES cross-attention (D)", "N": 547, "MAE": 0.8382, "RMSE": 1.0753, "Pearson": 0.7809, "Spearman": 0.7721},
    ])


def load_predictions(label):
    path = PREDICTION_FILES[label]
    frame = read_csv(path)
    if frame.empty:
        return frame
    # Normalize prediction columns from each evaluation script.
    if "actual_pAff" not in frame.columns:
        actual = next((c for c in frame.columns if "actual" in c.lower() and "paff" in c.lower()), None)
        if actual:
            frame = frame.rename(columns={actual: "actual_pAff"})
    pred = next((c for c in frame.columns if "predicted_paff" in c.lower()), None)
    if pred and pred != "predicted_pAff":
        frame = frame.rename(columns={pred: "predicted_pAff"})
    return frame


comparison = load_comparison()
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Model Comparison", "Predictions", "Dataset Overview", "Project Files"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Drug–Target Binding Affinity Predictor")
st.sidebar.caption("BindingDB • Viral Ki evaluation")

st.title("🧬 Drug–Target Binding Affinity Predictor")
st.markdown("Predicting drug–protein binding affinity using molecular graphs, SMILES, and protein sequences.")

if page == "Home":
    st.subheader("Project at a glance")
    best_mae = comparison.loc[comparison["MAE"].idxmin()] if not comparison.empty else None
    best_rmse = comparison.loc[comparison["RMSE"].idxmin()] if not comparison.empty else None
    pred_available = sum(path.exists() for path in PREDICTION_FILES.values())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models compared", len(comparison))
    c2.metric("Test examples", int(comparison["N"].max()) if "N" in comparison and comparison["N"].notna().any() else 547)
    c3.metric("Graph shards", "463")
    c4.metric("Prediction files found", f"{pred_available}/4")

    left, right = st.columns([1.6, 1])
    with left:
        st.subheader("Model performance")
        chart_data = comparison.melt(
            id_vars="Model", value_vars=["MAE", "RMSE"],
            var_name="Metric", value_name="Error",
        )
        fig = px.bar(
            chart_data, x="Model", y="Error", color="Metric", barmode="group",
            title="Error metrics (lower is better)", text_auto=".3f",
        )
        fig.update_layout(legend_title_text="", xaxis_title="", yaxis_title="Error in pAff units")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Quick interpretation")
        if best_mae is not None:
            st.info(f"**Lowest MAE:** {best_mae['Model']} ({best_mae['MAE']:.4f})")
        if best_rmse is not None:
            st.info(f"**Lowest RMSE:** {best_rmse['Model']} ({best_rmse['RMSE']:.4f})")
        st.markdown(
            "**Important:** these results are from one test split and one run per model. "
            "They are useful for project comparison, but do not by themselves prove generalization to unseen viral targets."
        )

    st.subheader("All-model summary")
    st.dataframe(comparison.round(4), use_container_width=True, hide_index=True)
    st.caption("Affinity label: pAff = 9 − log10(affinity in nM), for measurements represented in nM.")

elif page == "Model Comparison":
    st.subheader("Compare the four model variants")
    metric = st.selectbox("Choose a metric", ["MAE", "RMSE", "Pearson", "Spearman"])
    lower_better = metric in {"MAE", "RMSE"}
    chart = px.bar(
        comparison.sort_values(metric, ascending=lower_better),
        x="Model", y=metric, color="Model", text_auto=".4f",
        title=f"{metric} comparison ({'lower' if lower_better else 'higher'} is better)",
    )
    chart.update_layout(showlegend=False, xaxis_title="", yaxis_title=metric)
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(comparison.round(4), use_container_width=True, hide_index=True)
    st.download_button(
        "Download comparison CSV",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name="all_model_comparison.csv",
        mime="text/csv",
    )
    st.caption("Correlation measures ranking/association; MAE and RMSE measure prediction error.")

elif page == "Predictions":
    st.subheader("Inspect test-set predictions")
    model_name = st.selectbox("Select model", list(PREDICTION_FILES.keys()))
    predictions = load_predictions(model_name)
    if predictions.empty:
        st.warning(f"Prediction file not found or empty: {PREDICTION_FILES[model_name]}")
    elif "actual_pAff" not in predictions.columns or "predicted_pAff" not in predictions.columns:
        st.error("This prediction CSV does not have recognizable actual and predicted pAff columns.")
        st.dataframe(predictions.head(100), use_container_width=True)
    else:
        predictions["actual_pAff"] = pd.to_numeric(predictions["actual_pAff"], errors="coerce")
        predictions["predicted_pAff"] = pd.to_numeric(predictions["predicted_pAff"], errors="coerce")
        predictions = predictions.dropna(subset=["actual_pAff", "predicted_pAff"])
        predictions["absolute_error"] = (predictions["actual_pAff"] - predictions["predicted_pAff"]).abs()
        a, b, c = st.columns(3)
        a.metric("Rows", len(predictions))
        b.metric("MAE", f"{predictions['absolute_error'].mean():.4f}")
        c.metric("RMSE", f"{((predictions['actual_pAff'] - predictions['predicted_pAff']) ** 2).mean() ** 0.5:.4f}")
        scatter = px.scatter(
            predictions, x="actual_pAff", y="predicted_pAff",
            title="Predicted pAff vs actual pAff", opacity=0.65,
            hover_data=["absolute_error"],
        )
        low = min(predictions["actual_pAff"].min(), predictions["predicted_pAff"].min())
        high = max(predictions["actual_pAff"].max(), predictions["predicted_pAff"].max())
        scatter.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=dict(dash="dash"))
        scatter.update_layout(xaxis_title="Actual pAff", yaxis_title="Predicted pAff")
        st.plotly_chart(scatter, use_container_width=True)
        st.dataframe(predictions.head(500), use_container_width=True, hide_index=True)
        st.download_button(
            "Download displayed model predictions",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name=f"{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_predictions.csv",
            mime="text/csv",
        )
        st.caption("This page inspects saved test predictions. It does not yet run new inference for a newly entered molecule.")

elif page == "Dataset Overview":
    st.subheader("Processed BindingDB data")
    files = {
        "Cleaned data": DATA_DIR / "bindingdb_clean.csv",
        "Labeled data": DATA_DIR / "bindingdb_labeled.csv",
        "Train split": DATA_DIR / "train.csv",
        "Validation split": DATA_DIR / "val.csv",
        "Test split": DATA_DIR / "test.csv",
        "Viral train": DATA_DIR / "train_viral.csv",
        "Viral validation": DATA_DIR / "val_viral.csv",
        "Viral test": DATA_DIR / "test_viral.csv",
        "Viral train (graph-ready)": DATA_DIR / "train_viral_graph.csv",
        "Viral validation (graph-ready)": DATA_DIR / "val_viral_graph.csv",
        "Viral test (graph-ready)": DATA_DIR / "test_viral_graph.csv",
    }
    rows = []
    for name, path in files.items():
        if path.exists():
            try:
                # Read only the header and count lines without loading the full
                # multi-million-row files into memory.
                cols = list(pd.read_csv(path, nrows=0).columns)
                with path.open("rb") as handle:
                    count = sum(1 for _ in handle) - 1
                rows.append({"Dataset": name, "Rows": max(count, 0), "Columns": len(cols), "File": str(path.relative_to(PROJECT_ROOT))})
            except Exception:
                rows.append({"Dataset": name, "Rows": "—", "Columns": "—", "File": str(path.relative_to(PROJECT_ROOT))})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No processed CSV files were found. Check that this app is inside the project src folder.")
    st.info("The raw BindingDB file is large. This dashboard avoids loading the full raw TSV into memory.")
    summary_path = DATA_DIR / "target_summary.csv"
    summary = read_csv(summary_path)
    if not summary.empty:
        st.subheader("Target summary preview")
        st.dataframe(summary.head(20), use_container_width=True, hide_index=True)

elif page == "Project Files":
    st.subheader("Model checkpoints and outputs")
    sections = {
        "Model checkpoints": PROJECT_ROOT / "checkpoints",
        "Charts": RESULTS_DIR,
        "Logs": PROJECT_ROOT / "logs",
        "Processed data": DATA_DIR,
    }
    for heading, folder in sections.items():
        with st.expander(heading, expanded=(heading == "Model checkpoints")):
            if folder.exists():
                files = sorted(p for p in folder.iterdir() if p.is_file())
                if files:
                    st.write("\n".join(f"• {p.name}  ({p.stat().st_size / 1024:.1f} KB)" for p in files))
                else:
                    st.caption("Folder exists but has no files.")
            else:
                st.caption("Folder not found.")
    st.subheader("Prediction outputs")
    output_rows = [
        {"File": path.name, "Available": path.exists()}
        for path in PREDICTION_FILES.values()
    ]
    st.dataframe(pd.DataFrame(output_rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(
    '<div class="small-note">Academic research demo • BindingDB-derived affinity labels • '
    'Verify scientific conclusions against the evaluation protocol before reporting.</div>',
    unsafe_allow_html=True,
)
