import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Grid Stability ML Evaluation", layout="wide")

st.title("⚡ Electrical Grid Stability Classification")
st.markdown("**BITS Pilani M.Tech (DSE) — Machine Learning Assignment 2**")

st.sidebar.header("User Controls")

# Requirement a: Dataset upload option (CSV)
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

# Requirement b: Model selection dropdown
model_options = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Support Vector Machine"
]
selected_model_name = st.sidebar.selectbox("Select ML Model", model_options)

@st.cache_resource
def load_all_artifacts():
    scaler = joblib.load("model/scaler.joblib")
    loaded_models = {
        "Logistic Regression": joblib.load("model/logistic_regression.joblib"),
        "Decision Tree": joblib.load("model/decision_tree.joblib"),
        "KNN": joblib.load("model/knn.joblib"),
        "Naive Bayes": joblib.load("model/naive_bayes.joblib"),
        "Support Vector Machine": joblib.load("model/support_vector_machine.joblib")
    }
    return scaler, loaded_models

try:
    scaler, models = load_all_artifacts()
    model = models[selected_model_name]
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Data resolution
if uploaded_file is not None:
    df_eval = pd.read_csv(uploaded_file)
else:
    try:
        df_eval = pd.read_csv("test_data.csv")
        st.sidebar.info("Using default `test_data.csv` (2,000 instances).")
    except Exception as e:
        st.error(f"Could not load fallback `test_data.csv`: {e}")
        st.stop()

target_col = 'stabf' if 'stabf' in df_eval.columns else ('target' if 'target' in df_eval.columns else None)
if target_col is None:
    st.error("Uploaded CSV must contain either 'stabf' or 'target' column.")
    st.stop()

X_eval = df_eval.drop(columns=[c for c in ['stab', 'stabf', 'target'] if c in df_eval.columns])
if df_eval[target_col].dtype == object:
    y_eval = (df_eval[target_col] == 'unstable').astype(int)
else:
    y_eval = df_eval[target_col].astype(int)

# Scale features for distance/margin-based models
if selected_model_name in ["Logistic Regression", "KNN", "Support Vector Machine"]:
    X_eval_prep = scaler.transform(X_eval)
else:
    X_eval_prep = X_eval

y_pred = model.predict(X_eval_prep)
y_proba = model.predict_proba(X_eval_prep)[:, 1]

st.subheader("Test Data Preview")
st.dataframe(df_eval.head(6), use_container_width=True)

# Requirement c: Display of evaluation metrics
st.markdown("---")
st.subheader(f"Evaluation Metrics: {selected_model_name}")

acc = accuracy_score(y_eval, y_pred)
auc = roc_auc_score(y_eval, y_proba)
prec = precision_score(y_eval, y_pred, zero_division=0)
rec = recall_score(y_eval, y_pred, zero_division=0)
f1 = f1_score(y_eval, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_eval, y_pred)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Accuracy", f"{acc:.4f}")
m2.metric("AUC", f"{auc:.4f}")
m3.metric("Precision", f"{prec:.4f}")
m4.metric("Recall", f"{rec:.4f}")
m5.metric("F1 Score", f"{f1:.4f}")
m6.metric("MCC Score", f"{mcc:.4f}")

# Requirement d: Confusion matrix or classification report
st.markdown("---")
col_cm, col_cr = st.columns([1, 1])

with col_cm:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_eval, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Stable (0)', 'Unstable (1)'],
        yticklabels=['Stable (0)', 'Unstable (1)'],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with col_cr:
    st.subheader("Classification Report")
    report = classification_report(
        y_eval, y_pred,
        target_names=['Stable (0)', 'Unstable (1)']
    )
    st.text(report)
