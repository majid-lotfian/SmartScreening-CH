#  ---------------------------------------------------------------------------
#  CH-C vs CH-T detector (Stage-2)
#  ---------------------------------------------------------------------------
#  • Train set: 2018-2021 rows where CH3 ≠ 0
#  • Model: cost-sensitive CatBoost (fallback XGB) + isotonic calibration
#  • Grid-search positive weight  {1.0, 1.5, 2, 3, 4, 6}
#  • Threshold picked in CV:  Sens ≥ 0.93  &  PPV maximised
#  • Exports: metrics_*.csv , cm_*.png , predictions_*.xlsx
#  ---------------------------------------------------------------------------

import os, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------ #
# 1.  file locations                                                       #
# ------------------------------------------------------------------------ #
DATA_DIR   = "CH_exp1B_PreproccTotal_20_09_2024_10-20"
TRAIN_FILE = f"{DATA_DIR}/Data_train_2018_2021.csv"
TEST_FILE  = f"{DATA_DIR}/Data_test_2022.csv"
BATCH_FILE = f"{DATA_DIR}/Batch-CI1.csv"
OUT_DIR    = "new_results/stage2_cost_sensitive-CI4"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------------ #
# 2.  helpers                                                              #
# ------------------------------------------------------------------------ #
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ("Year", "CH"):
        if c in df: df.drop(columns=c, inplace=True)
    if "sex" in df: df["sex"] = df["sex"].map({"m": 0, "v": 1})
    return df

def export_cm(cm, tag):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion matrix – {tag}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/cm_{tag}.png")
    plt.close()

def metrics_from_cm(cm, y_true, y_pred, y_proba):
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if tp + fn else 0
    spec = tn / (tn + fp) if tn + fp else 0
    ppv  = tp / (tp + fp) if tp + fp else 0
    npv  = tn / (tn + fn) if tn + fn else 0
    acc  = accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average="macro")
    auc  = roc_auc_score(y_true, y_proba)
    return dict(Sensitivity=sens, Specificity=spec, PPV=ppv,
                NPV=npv, Accuracy=acc, **{"F1 Macro": f1m}, AUC=auc)

# ------------------------------------------------------------------------ #
# 3.  model factory (CatBoost → fallback XGB)                              #
# ------------------------------------------------------------------------ #
try:
    from catboost import CatBoostClassifier
    def make_model(pos_weight: float, seed: int):
        return CatBoostClassifier(
            depth=5, learning_rate=0.06, iterations=250,
            class_weights=[1.0, pos_weight],
            random_seed=seed, verbose=False,
            loss_function="Logloss", early_stopping_rounds=25
        )
except ModuleNotFoundError:
    from xgboost import XGBClassifier
    print("⚠️  CatBoost not installed – falling back to XGBoost.")
    def make_model(pos_weight: float, seed: int):
        return XGBClassifier(
            objective="binary:logistic", learning_rate=0.06,
            max_depth=5, n_estimators=250, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss",
            min_child_weight=4, scale_pos_weight=pos_weight,
            random_state=seed
        )

# ------------------------------------------------------------------------ #
# 4.  fit+calibrate helper                                                 #
# ------------------------------------------------------------------------ #
def fit_calibrated(X, y, pos_w, seed):
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    base = make_model(pos_w, seed)
    base.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    cal  = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    cal.fit(X_va, y_va)
    return cal

# ------------------------------------------------------------------------ #
# 5.  load train data, grid-search weight                                  #
# ------------------------------------------------------------------------ #
df_all   = load_csv(TRAIN_FILE)
df_ch    = df_all[df_all["CH3"] != 0].copy()           # CH only
X_all    = df_ch.drop(columns=["CH3", "kind"]).fillna(df_all.median())
y_all    = (df_ch["CH3"] == 2).astype(int)             # 1 = CH-C

print(f"Training rows: {len(y_all)}  (CH-C={y_all.sum()}, CH-T={len(y_all)-y_all.sum()})")

weights_to_try = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
best_w, best_thr, best_ppv = None, None, 0

skf = StratifiedKFold(5, shuffle=True, random_state=42)
print("\n→ Building out-of-fold probabilities (cost-sensitive)…")
for w in weights_to_try:
    oof_prob = np.zeros(len(y_all))
    for fold, (tr, va) in enumerate(skf.split(X_all, y_all), 1):
        model = fit_calibrated(X_all.iloc[tr], y_all.iloc[tr], pos_w=w, seed=1000+fold)
        oof_prob[va] = model.predict_proba(X_all.iloc[va])[:, 1]

    # threshold sweep at Sens≥0.93
    thr_local, ppv_local = None, 0
    for thr in np.arange(0.05, 0.51, 0.01):
        pred = (oof_prob >= thr).astype(int)
        sens = (pred & y_all).sum() / y_all.sum()
        if sens >= 0.93:
            ppv = (pred & y_all).sum() / pred.sum() if pred.sum() else 0
            if ppv > ppv_local:
                thr_local, ppv_local = thr, ppv
    print(f"  weight={w:<4}  best_thr={thr_local}  CV_PPv={ppv_local:.3f}")

    if ppv_local > best_ppv:
        best_w, best_thr, best_ppv = w, thr_local, ppv_local

print(f"\n✓ Selected pos_weight={best_w}  ϑ={best_thr:.2f}  (OOF PPV={best_ppv:.3f})")

# ------------------------------------------------------------------------ #
# 6.  train final calibrated model on *all* CH rows                        #
# ------------------------------------------------------------------------ #
print("\n→ Training final model …")
final_model = fit_calibrated(X_all, y_all, pos_w=best_w, seed=2025)

def predict_proba(X): return final_model.predict_proba(X)[:, 1]

# ------------------------------------------------------------------------ #
# 7.  evaluation & export                                                  #
# ------------------------------------------------------------------------ #
def evaluate(path, tag):
    df = load_csv(path)
    df = df[df["CH3"] != 0].copy()          # stage-2 only sees CH babies
    if df.empty:
        print(f"{tag}: no CH rows – skipped."); return

    X = df.drop(columns=["CH3", "kind"]).fillna(df_all.median())
    y = (df["CH3"] == 2).astype(int)
    p = predict_proba(X)
    yhat = (p >= best_thr).astype(int)

    cm = confusion_matrix(y, yhat)
    export_cm(cm, tag)
    m  = metrics_from_cm(cm, y, yhat, p)
    pd.DataFrame([m]).to_csv(f"{OUT_DIR}/metrics_{tag}.csv", index=False)
    print(f"{tag}: Sens={m['Sensitivity']:.3f}  PPV={m['PPV']:.3f}  Spec={m['Specificity']:.3f}")

    pd.DataFrame({
        "Patient ID": df["kind"], "Binary_pred": yhat, "GroundTruth": y
    }).to_excel(f"{OUT_DIR}/predictions_{tag}.xlsx", index=False)

print("\n→ 2022 hold-out")
evaluate(TEST_FILE, "2022_Holdout")

if os.path.exists(BATCH_FILE):
    print("→ Batch file")
    evaluate(BATCH_FILE, "Batch_Final")

print("\n✅ Stage-2 training & evaluation complete – outputs in", OUT_DIR)
