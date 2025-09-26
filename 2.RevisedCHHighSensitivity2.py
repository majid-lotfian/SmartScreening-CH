# CH Screening – Sens ≥ 0.99 with Calibrated Probabilities and CV‑based Threshold
# ---------------------------------------------------------------------------
# Workflow
# 1. Load 2018‑2021 data → 5‑fold CV → produce **out‑of‑fold calibrated probs** for every baby.
# 2. Find the **lowest threshold θ** that gives Sens ≥ 0.99 on the pooled CV probs while *maximising PPV*.
# 3. Retrain each model on the **full 2018‑2021** set, wrap in the same calibration method.
# 4. Soft‑vote (weighted) probabilities  → apply frozen θ  → evaluate on 2022 hold‑out + optional batch.
# 5. Export metrics + confusion matrices just like before.

import os, warnings, json, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def binarise_y(y):
    return np.where((y == 1) | (y == 2), 1, 0)


def metrics_and_cm(y_true_bin, y_pred_bin, y_proba_pos):
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    TN, FP, FN, TP = cm.ravel()
    sens = TP / (TP + FN) if TP + FN else 0
    spec = TN / (TN + FP) if TN + FP else 0
    ppv  = TP / (TP + FP) if TP + FP else 0
    npv  = TN / (TN + FN) if TN + FN else 0
    acc  = accuracy_score(y_true_bin, y_pred_bin)
    f1m  = f1_score(y_true_bin, y_pred_bin, average="macro")
    auc_ = roc_auc_score(y_true_bin, y_proba_pos)
    return {"Sensitivity":sens,"Specificity":spec,"PPV":ppv,"NPV":npv,"Accuracy":acc,"F1 Macro":f1m,"AUC":auc_}, cm


def save_metrics(tag, m, cm, outdir):
    pd.DataFrame([m]).to_csv(os.path.join(outdir, f"metrics_{tag}.csv"), index=False)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues"); plt.title(f"CM – {tag}"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"cm_{tag}.png")); plt.close()

# ---------------------------------------------------------------------------
# Load & preprocess
# ---------------------------------------------------------------------------

def load(path):
    df = pd.read_csv(path)
    for col in ("Year","CH"):
        df.drop(columns=[c for c in [col] if c in df.columns], inplace=True)
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"m":0,"v":1})
    return df

train_df = load("CH_exp1B_PreproccTotal_20_09_2024_10-20/Data_train_2018_2021.csv")
X_all = train_df.drop(columns=["CH3","kind"]).fillna(train_df.median())
y_all = train_df["CH3"].astype(int)

# Class‑weights
cls_cnt = y_all.value_counts().to_dict(); N=len(y_all)
class_w = {c: N/(len(cls_cnt)*cnt) for c,cnt in cls_cnt.items()}

base_models = {
    "RF": RandomForestClassifier(n_estimators=500,max_depth=50,min_samples_leaf=3,
                                 bootstrap=False,random_state=42,class_weight=class_w),
    "XGB": XGBClassifier(objective="multi:softprob",num_class=3,learning_rate=0.05,
                          max_depth=6,colsample_bytree=0.7,random_state=42,eval_metric="mlogloss"),
    "CAT": CatBoostClassifier(iterations=300,depth=6,learning_rate=0.1,verbose=0,
                               random_seed=42,class_weights=[class_w.get(i,1) for i in range(3)])
}

# ---------------------------------------------------------------------------
# 5‑fold CV → get calibrated out‑of‑fold probs
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
proba_oof = np.zeros((len(y_all),3))
for train_idx, val_idx in cv.split(X_all,y_all):
    X_tr, X_va = X_all.iloc[train_idx], X_all.iloc[val_idx]
    y_tr, y_va = y_all.iloc[train_idx], y_all.iloc[val_idx]
    probs_fold = []
    for name, mdl in base_models.items():
        mdl_clone = mdl.__class__(**mdl.get_params())
        if name=="XGB":
            mdl_clone.fit(X_tr,y_tr,sample_weight=y_tr.map(class_w))
        else:
            mdl_clone.fit(X_tr,y_tr)
        cal = CalibratedClassifierCV(mdl_clone,method="isotonic",cv="prefit")
        cal.fit(X_va,y_va)
        probs_fold.append(cal.predict_proba(X_va))
    proba_oof[val_idx] = np.mean(probs_fold,axis=0)

pos_oof = proba_oof[:,1]+proba_oof[:,2]
y_oof_bin = binarise_y(y_all)

# Threshold search
best_thr,best_ppv = None,0
for thr in np.arange(0.05,0.51,0.01):
    pred = (pos_oof>=thr).astype(int)
    sens = (pred & y_oof_bin).sum()/y_oof_bin.sum()
    if sens>=0.99:
        ppv  = (pred & y_oof_bin).sum()/pred.sum() if pred.sum() else 0
        if ppv>best_ppv:
            best_ppv,best_thr = ppv,thr
# fallback
if best_thr is None:
    best_thr = max(np.arange(0.05,0.51,0.01), key=lambda t: ((pos_oof>=t)&y_oof_bin).sum()/y_oof_bin.sum())
print(f"Chosen θ={best_thr:.2f}  (OOF Sens={(pos_oof>=best_thr).astype(int).dot(y_oof_bin)/y_oof_bin.sum():.3f}  PPV≈{best_ppv:.3f})")

# ---------------------------------------------------------------------------
# Retrain full models + calibration wrapper
# ---------------------------------------------------------------------------
calibrated_models = {}
for name, mdl in base_models.items():
    if name=="XGB":
        mdl.fit(X_all,y_all,sample_weight=y_all.map(class_w))
    else:
        mdl.fit(X_all,y_all)
    calibrated_models[name] = CalibratedClassifierCV(mdl,method="isotonic",cv=3)
    calibrated_models[name].fit(X_all,y_all)

def ens_proba(X):
    return np.mean([m.predict_proba(X) for m in calibrated_models.values()],axis=0)

# ---------------------------------------------------------------------------
# Evaluation on hold‑out 2022 + optional batch
# ---------------------------------------------------------------------------
OUTDIR = os.path.join("new_results","sens99_calib_run2"); os.makedirs(OUTDIR,exist_ok=True)

def evaluate(file_path, tag):
    df = load(file_path)
    X = df.drop(columns=["CH3","kind"]).fillna(train_df.median())
    y = df["CH3"].astype(int)
    proba = ens_proba(X)
    pos_p = proba[:,1]+proba[:,2]
    preds = (pos_p>=best_thr).astype(int)
    m,cm = metrics_and_cm(binarise_y(y),preds,pos_p)
    save_metrics(tag,m,cm,OUTDIR)
    print(f"{tag}: Sens={m['Sensitivity']:.3f}  PPV={m['PPV']:.3f}  Spec={m['Specificity']:.3f}")

evaluate("CH_exp1B_PreproccTotal_20_09_2024_10-20/Data_test_2022.csv","2022_Holdout")
BATCH_FILE = "CH_exp1B_PreproccTotal_20_09_2024_10-20/Batch2 Data_test_2022.csv"
if os.path.exists(BATCH_FILE):
    evaluate(BATCH_FILE,"Batch_Final")

print("✅ All metrics & plots saved to", OUTDIR)
