# Two‑Stage CH Screening Pipeline (High‑Recall + Subtype)
# ======================================================
# Stage‑1 : Ensemble (RF + XGB + CatBoost) – 3‑class soft‑prob → binary screen
# Stage‑2 : Logistic Regression – distinguish primary (1) vs central (2) *only on referred babies*
# ------------------------------------------------------------------------------
# Training data  : 2018‑2021  (Data_train_2018_2021.csv)
# Hold‑out test  : 2022       (Data_test_2022.csv)
# Optional batch : Batch2 Data_test_2022.csv
# Outputs        : new_results/two_stage_run/*  (metrics CSV + CM PNGs)

import os, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
warnings.filterwarnings("ignore")

# -------------------------- helpers --------------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ("Year", "CH"):
        if c in df: df.drop(columns=c, inplace=True)
    if "sex" in df: df["sex"] = df["sex"].map({"m": 0, "v": 1})
    return df

def binarise(y: pd.Series) -> pd.Series:
    return (y != 0).astype(int)

def scr_metrics(y_true_bin, y_pred_bin, y_proba_pos):
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    TN, FP, FN, TP = cm.ravel()
    sens = TP / (TP + FN) if TP + FN else 0
    spec = TN / (TN + FP) if TN + FP else 0
    ppv  = TP / (TP + FP) if TP + FP else 0
    npv  = TN / (TN + FN) if TN + FN else 0
    acc  = accuracy_score(y_true_bin, y_pred_bin)
    f1   = f1_score(y_true_bin, y_pred_bin)
    auc_ = roc_auc_score(y_true_bin, y_proba_pos)
    return {"Sensitivity":sens,"Specificity":spec,"PPV":ppv,"NPV":npv,"Accuracy":acc,"F1":f1,"AUC":auc_}, cm

def subtype_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[1,2])
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    return {"Subtype Acc":acc,"Subtype F1":f1}, cm

def save_cm(cm, tag, outdir):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues"); plt.title(tag); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"cm_{tag}.png")); plt.close()

# -------------------- Stage‑1 training ----------------------
TRAIN = "CH_exp1B_PreproccTotal_20_09_2024_10-20/Data_train_2018_2021.csv"
train_df = load_csv(TRAIN)
X_all = train_df.drop(columns=["CH3", "kind"]).fillna(train_df.median())
y_all = train_df["CH3"].astype(int)

# class weights for RF & Cat
cls_counts = y_all.value_counts(); N_tot = len(y_all)
class_w = {c: N_tot/(len(cls_counts)*cnt) for c, cnt in cls_counts.items()}

base_models = {
    "RF":  RandomForestClassifier(n_estimators=500, max_depth=50, min_samples_leaf=3, bootstrap=False, random_state=42, class_weight=class_w),
    "XGB": XGBClassifier(objective="multi:softprob", num_class=3, learning_rate=0.05, max_depth=6, colsample_bytree=0.7, random_state=42, eval_metric="mlogloss"),
    "CAT": CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, random_seed=42, verbose=0, class_weights=[class_w.get(i,1) for i in range(3)])
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
proba_oof = np.zeros((len(y_all), 3))
for tr_idx, va_idx in cv.split(X_all, y_all):
    X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
    y_tr, y_va = y_all.iloc[tr_idx], y_all.iloc[va_idx]
    probs_fold = []
    for name, mdl in base_models.items():
        model = mdl.__class__(**mdl.get_params())
        if name == "XGB":
            model.fit(X_tr, y_tr, sample_weight=y_tr.map(class_w))
        else:
            model.fit(X_tr, y_tr)
        cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        cal.fit(X_va, y_va)
        probs_fold.append(cal.predict_proba(X_va))
    proba_oof[va_idx] = np.mean(probs_fold, axis=0)

pos_oof = proba_oof[:,1] + proba_oof[:,2]
y_bin_oof = binarise(y_all)

# θ₁ search for Sens ≥ 0.99
best_thr, best_ppv = None, 0
for t in np.arange(0.01, 0.51, 0.01):
    pred = (pos_oof >= t).astype(int)
    sens = (pred & y_bin_oof).sum() / y_bin_oof.sum()
    if sens >= 0.99:
        ppv = (pred & y_bin_oof).sum() / pred.sum() if pred.sum() else 0
        if ppv > best_ppv: best_thr, best_ppv = t, ppv
if best_thr is None:
    best_thr = 0.05  # fallback
print(f"Stage‑1 θ₁ = {best_thr:.2f}  (OOF Sens≥0.99, PPV≈{best_ppv:.3f})")

# -------------------- Retrain full Stage‑1 models --------------------
cal_models = {}
for name, mdl in base_models.items():
    if name == "XGB":
        mdl.fit(X_all, y_all, sample_weight=y_all.map(class_w))
    else:
        mdl.fit(X_all, y_all)
    cal = CalibratedClassifierCV(mdl, method="isotonic", cv=3)
    cal.fit(X_all, y_all)
    cal_models[name] = cal

def stage1_proba(X):
    return np.mean([m.predict_proba(X) for m in cal_models.values()], axis=0)

# -------------------- Stage‑2 training (primary vs central) --------
referred_mask = pos_oof >= best_thr
X_ref = X_all[referred_mask]
y_ref = y_all[referred_mask]
mask_pc = y_ref.isin([1, 2])
X_pc = X_ref[mask_pc]
sub_y = (y_ref[mask_pc] == 2).astype(int)  # 0=primary, 1=central

stage2_base = LogisticRegression(max_iter=200, class_weight="balanced")
cal2 = CalibratedClassifierCV(stage2_base, method="isotonic", cv=5)
cal2.fit(X_pc, sub_y)
joblib.dump(cal2, "stage2_subtype.pkl")

# ------------------------ Evaluation ------------------------
OUTDIR = "new_results/two_stage_run"; os.makedirs(OUTDIR, exist_ok=True)

def evaluate(file_path: str, tag: str):
    df = load_csv(file_path)
    X = df.drop(columns=["CH3", "kind"]).fillna(train_df.median())
    y_true = df["CH3"].astype(int)
    proba = stage1_proba(X)
    pos_p = proba[:,1] + proba[:,2]
    screen_pred = (pos_p >= best_thr).astype(int)

    # --- Stage‑1 metrics ---
    m1, cm1 = scr_metrics(binarise(y_true), screen_pred, pos_p)
    pd.DataFrame([m1]).to_csv(os.path.join(OUTDIR, f"metrics_screen_{tag}.csv"), index=False)
    save_cm(cm1, f"Screen CM – {tag}", OUTDIR)

    # --- Stage‑2 subtype only on referred ---
    referred_idx = screen_pred.astype(bool)
    if referred_idx.sum():
        X_ref = X[referred_idx]
        y_ref = y_true[referred_idx]
        mask_pc = y_ref.isin([1, 2])
        X_pc_eval = X_ref[mask_pc]
        y_pc_true = y_ref[mask_pc]
        if len(y_pc_true):
            p_cent = cal2.predict_proba(X_pc_eval)[:,1]
            subtype_pred = np.where(p_cent >= 0.5, 2, 1)
            m2, cm2 = subtype_metrics(y_pc_true, subtype_pred)
            pd.DataFrame([m2]).to_csv(os.path.join(OUTDIR, f"metrics_subtype_{tag}.csv"), index=False)
            save_cm(cm2, f"Subtype CM – {tag}", OUTDIR)
            print(f"{tag}  Screen Sens={m1['Sensitivity']:.3f}  PPV={m1['PPV']:.3f} | Subtype ACC={m2['Subtype Acc']:.3f}")
        else:
            print(f"{tag}  No primary/central cases in referred set → subtype eval skipped")
    else:
        print(f"{tag}  No babies crossed θ₁ (unlikely) – check θ₁ setting")

TEST_22 = "CH_exp1B_PreproccTotal_20_09_2024_10-20/Data_test_2022.csv"
BATCH   = "CH_exp1B_PreproccTotal_20_09_2024_10-20/Batch2 Data_test_2022.csv"

evaluate(TEST_22, "2022_Holdout")
if os.path.exists(BATCH): evaluate(BATCH, "Batch_Final")

print("✅ Two‑stage pipeline finished – results in", OUTDIR)
