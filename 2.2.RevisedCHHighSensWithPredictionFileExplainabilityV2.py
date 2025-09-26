# Soft‑Ensemble Screening (RF+XGB+CAT) – Adds patient‑level .xlsx output
# ----------------------------------------------------------------------

import os, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, shap
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
warnings.filterwarnings("ignore")

# -------------------- helpers --------------------

def binarise(y):
    return (y != 0).astype(int)

def metrics(cm, y_true_bin, y_pred_bin, y_proba_pos):
    TN, FP, FN, TP = cm.ravel()
    sens = TP/(TP+FN) if TP+FN else 0
    spec = TN/(TN+FP) if TN+FP else 0
    ppv  = TP/(TP+FP) if TP+FP else 0
    npv  = TN/(TN+FN) if TN+FN else 0
    acc  = accuracy_score(y_true_bin, y_pred_bin)
    f1m  = f1_score(y_true_bin, y_pred_bin, average="macro")
    auc_ = roc_auc_score(y_true_bin, y_proba_pos)
    return {"Sensitivity":sens,"Specificity":spec,"PPV":ppv,"NPV":npv,"Accuracy":acc,"F1 Macro":f1m,"AUC":auc_}

def save_cm(cm, tag, outdir):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues"); plt.title(tag); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"cm_{tag}.png")); plt.close()

# -------------------- load --------------------

def load(path):
    df = pd.read_csv(path)
    for c in ("Year","CH"):
        if c in df: df.drop(columns=c, inplace=True)
    if "sex" in df: df["sex"] = df["sex"].map({"m":0,"v":1})
    return df

TRAIN = "CH_exp1B_PreproccTotal_20_09_2024_10-20/Data_train_2018_2021.csv"
train_df = load(TRAIN)
X_all = train_df.drop(columns=["CH3","kind"]).fillna(train_df.median())
y_all = train_df["CH3"].astype(int)

# class weights for RF & CAT
cnt = y_all.value_counts(); N=len(y_all)
cls_w = {c: N/(len(cnt)*cntc) for c,cntc in cnt.items()}

models = {
    "RF": RandomForestClassifier(n_estimators=500, max_depth=50, min_samples_leaf=3, bootstrap=False, random_state=42, class_weight=cls_w),
    "XGB": XGBClassifier(objective="multi:softprob", num_class=3, learning_rate=0.05, max_depth=6, colsample_bytree=0.7, random_state=42, eval_metric="mlogloss"),
    "CAT": CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=0, random_seed=42, class_weights=[cls_w.get(i,1) for i in range(3)])
}

# -------------------- CV for θ --------------------
cv = StratifiedKFold(5, shuffle=True, random_state=42)
proba_oof = np.zeros((len(y_all),3))
for tr,va in cv.split(X_all,y_all):
    X_tr,X_va = X_all.iloc[tr], X_all.iloc[va]
    y_tr,y_va = y_all.iloc[tr], y_all.iloc[va]
    fold_probs=[]
    for name,mdl in models.items():
        m=mdl.__class__(**mdl.get_params())
        if name=="XGB":
            m.fit(X_tr,y_tr,sample_weight=y_tr.map(cls_w))
        else:
            m.fit(X_tr,y_tr)
        cal=CalibratedClassifierCV(m,method="isotonic",cv="prefit")
        cal.fit(X_va,y_va)
        fold_probs.append(cal.predict_proba(X_va))
    proba_oof[va]=np.mean(fold_probs,axis=0)

pos_oof = proba_oof[:,1]+proba_oof[:,2]
y_bin = binarise(y_all)

best_thr,best_ppv=None,0
for t in np.arange(0.05,0.51,0.01):
    pred=(pos_oof>=t).astype(int); sens=(pred&y_bin).sum()/y_bin.sum()
    if sens>=0.99:
        ppv=(pred&y_bin).sum()/pred.sum() if pred.sum() else 0
        if ppv>best_ppv: best_thr,best_ppv=t,ppv
if best_thr is None: best_thr=0.05
print(f"θ₁={best_thr:.2f}  (OOF Sens≥0.99, PPV≈{best_ppv:.3f})")

# -------------------- retrain full calibrated models --------------------
cal_models={}
for name,mdl in models.items():
    if name=="XGB": mdl.fit(X_all,y_all,sample_weight=y_all.map(cls_w))
    else: mdl.fit(X_all,y_all)
    cal=CalibratedClassifierCV(mdl,method="isotonic",cv=3)
    cal.fit(X_all,y_all)
    cal_models[name]=cal

ens=lambda X: np.mean([m.predict_proba(X) for m in cal_models.values()],axis=0)

# -------------------- evaluation + XLSX export --------------------
OUT="new_results/soft_ens_with_preds_explain_v2"; os.makedirs(OUT,exist_ok=True)

def evaluate(filepath, tag):
    df=load(filepath)
    X=df.drop(columns=["CH3","kind"]).fillna(train_df.median())
    y=df["CH3"].astype(int)
    proba=ens(X)
    pos_p=proba[:,1]+proba[:,2]
    bin_pred=(pos_p>=best_thr).astype(int)
    multi_pred=proba.argmax(axis=1)

    # metrics
    cm=confusion_matrix(binarise(y),bin_pred)
    m=metrics(cm,binarise(y),bin_pred,pos_p)
    pd.DataFrame([m]).to_csv(os.path.join(OUT,f"metrics_{tag}.csv"),index=False)
    save_cm(cm,f"Screen CM – {tag}",OUT)
    print(f"{tag}: Sens={m['Sensitivity']:.3f} PPV={m['PPV']:.3f} Spec={m['Specificity']:.3f}")

    # patient-level excel
    result=pd.DataFrame({
        "Patient ID":df["kind"],
        "Binary_pred":bin_pred,
        "Multi_pred":multi_pred,
        "GroundTruth_CH3":y
    })
    result.to_excel(os.path.join(OUT,f"predictions_{tag}.xlsx"),index=False)




TEST="CH_exp1B_PreproccTotal_20_09_2024_10-20/Data_test_2022.csv"
BATCH="CH_exp1B_PreproccTotal_20_09_2024_10-20/Batch2 Data_test_2022.csv"

evaluate(TEST,"2022_Holdout")
if os.path.exists(BATCH): evaluate(BATCH,"Batch_Final")

# -------------------- KERNEL EXPLAINER FOR ENSEMBLE --------------------

print("🔍 Explaining ensemble predictions using SHAP KernelExplainer...")

# Define ensemble prediction function
def ensemble_predict(X):
    if isinstance(X, pd.DataFrame):
        return np.mean([model.predict_proba(X) for model in cal_models.values()], axis=0)
    else:
        X_df = pd.DataFrame(X, columns=X_all.columns)
        return np.mean([model.predict_proba(X_df) for model in cal_models.values()], axis=0)

# Background sample (small, representative)
X_background = shap.sample(X_all, 100, random_state=42)

# Subset of data to explain (keep it small for performance)
#X_explain = X_all.sample(n=20, random_state=123)

df_test = load(TEST)  # or your path to test file
df_test = df_test[df_test["CH3"] != 0].copy()
X_test = df_test.drop(columns=["CH3", "kind"]).fillna(train_df.median())

X_explain = X_test.copy()  # use all CH rows in the test set 


# Create explainer
explainer = shap.KernelExplainer(ensemble_predict, X_background)

# Compute SHAP values (slow)
shap_values = explainer.shap_values(X_explain, nsamples="auto")

print("Type of shap_values:", type(shap_values))
if isinstance(shap_values, list):
    print("Shape of shap_values[1]:", np.array(shap_values[1]).shape)
else:
    print("Shape of shap_values:", np.array(shap_values).shape)
print("Shape of X_explain:", X_explain.shape)


# -------------------- SHAP Plot for POSITIVE Class (Class 1 + 2) --------------------

print("🔍 Plotting SHAP summary for POSITIVE class (Class 1 + 2)...")

# Case: multiclass SHAP output as a single ndarray (n_samples, n_features, n_classes)
if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_pos = shap_values[:, :, 1] + shap_values[:, :, 2]
    shap.summary_plot(shap_pos, X_explain, show=False)

# Case: shap_values returned as list (one array per class)
elif isinstance(shap_values, list) and len(shap_values) == 3:
    shap_pos = shap_values[1] + shap_values[2]
    shap.summary_plot(shap_pos, X_explain, show=False)

# Fallback
else:
    print("⚠️ Unexpected SHAP shape. Plotting as-is.")
    shap.summary_plot(shap_values, X_explain, show=False)

# Save plot
plt.title("SHAP Summary – KernelExplainer (Ensemble, Class 1 + 2)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "kernel_shap_summary_positive.png"))
plt.close()

print("✅ SHAP summary for Class 1 + 2 saved as:", os.path.join(OUT, "kernel_shap_summary_positive.png"))

# Convert SHAP values to DataFrame
shap_df = pd.DataFrame(shap_pos, columns=X_explain.columns)
shap_df.insert(0, "PatientIndex", X_explain.index)  # optional: retain patient ID

# Save SHAP values to CSV
shap_csv_path = os.path.join(OUT, "kernel_shap_values_positive.csv")
shap_df.to_csv(shap_csv_path, index=False)

print("✅ SHAP values saved to:", shap_csv_path)