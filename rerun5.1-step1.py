import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings("ignore")

# === 0. Setup output directory ===
output_dir = os.path.join("trained_models")
os.makedirs(output_dir, exist_ok=True)

# === 1. Load training/validation dataset ===
data_path = "CH_exp1B_PreproccTotal_20_09_2024_10-20/Data_train_2018_2021.csv"
df = pd.read_csv(data_path)

# Drop unnecessary columns
if "Year" in df.columns:
    df.drop(columns=["Year"], inplace=True)

# Encode 'sex'
if "sex" in df.columns:
    df["sex"] = df["sex"].map({"m": 0, "v": 1})

# === 2. Define features and target ===
X = df.drop(columns=["CH3", "CH"] if "CH" in df.columns else ["CH3"])
y = df["CH3"]

# === 3. Split train/val ===
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === 4. SMOTE Oversampling ===
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# === 5. Compute class weights ===
unique_classes = np.unique(y_train_res)
weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train_res)
class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, weights)}

# === 6. Define models ===
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=500, max_depth=50, min_samples_leaf=3, bootstrap=False,
        class_weight=class_weight_dict, random_state=42
    ),
    "XGBoost": XGBClassifier(
        learning_rate=0.05, max_depth=6, min_child_weight=1,
        colsample_bytree=0.7, use_label_encoder=False, eval_metric="mlogloss",
        random_state=42
    ),
    "CatBoost": CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.1,
        verbose=0, random_seed=42
    )
}

# === 7. Train and save models ===
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_res, y_train_res)
    model_path = os.path.join(output_dir, f"{name}_model.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Saved {name} to {model_path}")

print("\n🎉 All models trained and saved.")
