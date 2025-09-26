import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score,
    roc_curve, auc, accuracy_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize
import numpy as np
import shap
import warnings
warnings.filterwarnings("ignore")

# Output directory setup
output_dir = os.path.join("new_results", "ensemble_run5.1")
os.makedirs(output_dir, exist_ok=True)

# Load datasets
train_val_df = pd.read_csv("CH_exp1B_PreproccTotal_20_09_2024_10-20/Data_train_2018_2021.csv")
test_2022_df = pd.read_csv("CH_exp1B_PreproccTotal_20_09_2024_10-20/Data_test_2022.csv")

# Drop unneeded columns except 'kind' (Patient ID)
for df in [train_val_df, test_2022_df]:
    df.drop(columns=[col for col in ["Year", "CH"] if col in df.columns], inplace=True)

# Encode 'sex' feature
for df in [train_val_df, test_2022_df]:
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"m": 0, "v": 1})

# Split training/validation set
X = train_val_df.drop(columns=["CH3", "kind"])
y = train_val_df["CH3"]
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Balance training data with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Define models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=500, max_depth=50, min_samples_leaf=3, bootstrap=False, random_state=42),
    "XGBoost": XGBClassifier(learning_rate=0.05, max_depth=6, colsample_bytree=0.7, random_state=42, use_label_encoder=False, eval_metric="mlogloss"),
    "CatBoost": CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=0, random_seed=42)
}

# Train models and store predictions
trained_models = {}
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    trained_models[name] = model

# Ensemble prediction function
def ensemble_predict(X):
    # Drop unnecessary columns if they exist
    for col in ["Year", "CH"]:
        if col in X.columns:
            X = X.drop(columns=[col])
    # Ensure consistent columns with training data
    X = X[X_train.columns]
    # Encode 'sex' feature if needed
    if 'sex' in X.columns:
        X['sex'] = X['sex'].map({'m': 0, 'v': 1}).fillna(X['sex'])
    preds = []
    for model in trained_models.values():
        try:
            pred = model.predict(X)
            # If prediction is probabilistic, convert to class labels
            if len(pred.shape) > 1:
                pred = np.argmax(pred, axis=1)
            preds.append(pred)
        except AttributeError:
            print(f"Model {model} does not support predict. Skipping.")
            continue
    if len(preds) == 0:
        raise ValueError("No models with predict available.")
    # Stack predictions and apply majority voting
    preds = np.array(preds)
    final_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)
    return final_pred

# Ensemble prediction with probability function
def ensemble_predict_proba(X):
    # Drop unnecessary columns if they exist
    for col in ["Year", "CH"]:
        if col in X.columns:
            X = X.drop(columns=[col])
    # Ensure consistent columns with training data
    X = X[X_train.columns]
    # Encode 'sex' feature if needed
    if 'sex' in X.columns:
        X['sex'] = X['sex'].map({'m': 0, 'v': 1}).fillna(X['sex'])
    probas = []
    for model in trained_models.values():
        try:
            proba = model.predict_proba(X)
            probas.append(proba)
        except AttributeError:
            print(f"Model {model} does not support predict_proba. Skipping.")
            continue
    if len(probas) == 0:
        raise ValueError("No models with predict_proba available.")
    # Calculate the average probability across all models that support it
    avg_proba = np.mean(probas, axis=0)
    print("✅ Ensemble prediction with probability function updated to handle column consistency and unnecessary columns correctly.")
    return avg_proba

# Metrics calculation function
def calculate_metrics(y_true, y_pred, name):
    # Ensure that y_pred contains discrete class labels
    if len(y_pred.shape) > 1:  # If it is a probability array
        y_pred = np.argmax(y_pred, axis=1)
    
    # True Positives: Correctly predicted CH cases (1 or 2)
    TP = np.sum(((y_true == 1) | (y_true == 2)) & ((y_pred == 1) | (y_pred == 2)))
    # False Positives: Predicted as CH (1 or 2) but true label is no-CH (0)
    FP = np.sum((y_true == 0) & ((y_pred == 1) | (y_pred == 2)))

    # Calculate PPV, handling division by zero
    if TP + FP == 0:
        ppv = 0
    else:
        ppv = TP / (TP + FP)  # PPV calculation
    
    metrics = {
        "PPV": ppv,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1": f1_score(y_true, y_pred, average="weighted"),
        "AUC": roc_auc_score(label_binarize(y_true, classes=[0, 1, 2]), label_binarize(y_pred, classes=[0, 1, 2]), multi_class="ovr"),
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted")
    }

    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, f"ensemble_metrics_{name}.csv"), index=False)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - Ensemble ({name})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ensemble_conf_matrix_{name}.png"))
    plt.close()
    print("✅ Metrics calculation function correctly calculates PPV for CH cases.")


# Permutation importance calculation function
def compute_permutation_importance(X, y, model, name):
    result = permutation_importance(model, X, y, scoring="f1_macro", n_repeats=10, random_state=42)
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": result.importances_mean
    }).sort_values(by="Importance", ascending=False)
    importance_df.to_csv(os.path.join(output_dir, f"feature_importance_{name}.csv"), index=False)
    plt.figure(figsize=(8, 10))
    sns.barplot(data=importance_df.head(30), x="Importance", y="Feature", palette="viridis")
    plt.title(f"Top 30 Feature Importances - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"feature_importance_{name}.png"))
    plt.close()
    
    print("✅ Permutation importance calculation function loaded.")

# ROC Curve function
def plot_roc_curve(y_true, y_proba, name):
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(8, 6))
    colors = ['darkorange', 'green', 'steelblue']
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Ensemble ({name})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"roc_curve_{name}.png"))
    plt.close()

# Evaluate ensemble on 2022 dataset
X_2022 = test_2022_df.drop(columns=["CH3", "kind"])
y_2022 = test_2022_df["CH3"]
y_pred = ensemble_predict(X_2022)
y_proba = np.mean([model.predict_proba(X_2022) for model in trained_models.values()], axis=0)
calculate_metrics(y_2022, y_pred, "2022")
compute_permutation_importance(X_2022, y_2022, trained_models["RandomForest"], "2022")
plot_roc_curve(y_2022, y_proba, "2022")

# Batch processing

def batch_predict():
    batch_data = pd.read_csv("CH_exp1B_PreproccTotal_20_09_2024_10-20/Batch2 Data_test_2022.csv")
    # Drop unnecessary columns if they exist
    for col in ["Year", "CH"]:
        if col in batch_data.columns:
            batch_data = batch_data.drop(columns=[col])
    # Encode 'sex' feature if needed
    if 'sex' in batch_data.columns:
        batch_data['sex'] = batch_data['sex'].map({'m': 0, 'v': 1}).fillna(batch_data['sex'])
    patient_ids = batch_data["kind"]
    ground_truth = batch_data["CH3"].values.flatten()
    X_batch = batch_data.drop(columns=["CH3", "kind"])
    # Align columns
    X_batch = X_batch[X_train.columns]
    predictions = ensemble_predict(X_batch).flatten()
    y_proba_batch = ensemble_predict_proba(X_batch)
    calculate_metrics(ground_truth, predictions, "batch")
    compute_permutation_importance(X_batch, ground_truth, trained_models["RandomForest"], "batch")
    plot_roc_curve(ground_truth, y_proba_batch, "batch")
    result_df = pd.DataFrame({
        "Patient ID": patient_ids,
        "Prediction": predictions,
        "Ground Truth": ground_truth
    })
    output_file = os.path.join(output_dir, "batch_predictions.xlsx")
    result_df.to_excel(output_file, index=False)
    print(f"✅ Batch predictions saved to {output_file}")

# Run batch prediction
batch_predict()

print("✅ All outputs (metrics, confusion matrices, feature importances, ROC curves) generated.")