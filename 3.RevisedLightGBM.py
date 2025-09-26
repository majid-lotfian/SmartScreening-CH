# CH Screening – LightGBM Binary (v2)  
# Tuned for more stable splits & higher PPV while keeping high sensitivity
# -----------------------------------------------------------------------
# Key changes versus previous version
# • Explicit `metric='auc'` – guarantees positive gain evaluation
# • `min_gain_to_split = 0.0`   → allow tiny gains
# • `min_data_in_leaf   = 5`    → permit smaller leaves (previous default 20)
# • More boosting rounds (n_estimators = 800) + slightly higher LR = 0.05
# • Light bagging / feature‑fraction for generalisation
# • Same 5‑fold CV + isotonic calibration + threshold search (Sens ≥ 0.99)
# • Everything else (column cleaning, metric export) unchanged

import os, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
warnings.filterwarnings("ignore")

# ----------------- helper functions -----------------
BAD_CHARS='[]{}":,'; TRANS=str.maketrans({c:'_' for c in BAD_CHARS})

def clean_cols(df):
    return df.rename(columns=lambda c: str(c).translate(TRANS))

def load(path):
    df=pd.read_csv(path)
    for c in ("Year","CH"):
        if c in df: df.drop(columns=c,inplace=True)
    if 'sex' in df: df['sex']=df['sex'].map({'m':0,'v':1})
    return clean_cols(df)

def binarise(y):
    return (y!=0).astype(int)

def metric_dict(y_true,y_pred,y_prob):
    cm=confusion_matrix(y_true,y_pred); TN,FP,FN,TP=cm.ravel()
    sens=TP/(TP+FN) if TP+FN else 0
    spec=TN/(TN+FP) if TN+FP else 0
    ppv =TP/(TP+FP) if TP+FP else 0
    npv =TN/(TN+FN) if TN+FN else 0
    return {"Sensitivity":sens,"Specificity":spec,"PPV":ppv,"NPV":npv,
            "Accuracy":accuracy_score(y_true,y_pred),
            "F1":f1_score(y_true,y_pred),
            "AUC":roc_auc_score(y_true,y_prob)}, cm

def save(tag,m,cm,outdir):
    pd.DataFrame([m]).to_csv(os.path.join(outdir,f"metrics_{tag}.csv"),index=False)
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues'); plt.title(f"CM – {tag}"); plt.tight_layout()
    plt.savefig(os.path.join(outdir,f"cm_{tag}.png")); plt.close()

# ------------- load development data -------------
train_df=load("CH_exp1B_PreproccTotal_20_09_2024_10-20/Data_train_2018_2021.csv")
X_all=train_df.drop(columns=["CH3","kind"]).fillna(train_df.median())
y_all=binarise(train_df["CH3"].astype(int))

pos_weight=(len(y_all)-y_all.sum())/y_all.sum()

lgb_params=dict(
    objective='binary', metric='auc',
    scale_pos_weight=pos_weight,
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    min_gain_to_split=0.0,
    min_data_in_leaf=5,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    n_estimators=800,
    random_state=42
)

# ---------- 5‑fold CV with isotonic calibration ----------
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
oof_prob=np.zeros(len(y_all))
for tr,va in cv.split(X_all,y_all):
    model=lgb.LGBMClassifier(**lgb_params)
    model.fit(X_all.iloc[tr],y_all.iloc[tr])
    cal=CalibratedClassifierCV(model,method='isotonic',cv='prefit')
    cal.fit(X_all.iloc[va],y_all.iloc[va])
    oof_prob[va]=cal.predict_proba(X_all.iloc[va])[:,1]

best_thr,best_ppv,best_sens=None,0,0
for t in np.arange(0.01,0.51,0.01):
    p=(oof_prob>=t).astype(int); TP=((p==1)&(y_all==1)).sum(); FN=((p==0)&(y_all==1)).sum(); FP=((p==1)&(y_all==0)).sum()
    sens=TP/(TP+FN) if TP+FN else 0
    if sens>=0.99:
        ppv=TP/(TP+FP) if TP+FP else 0
        if ppv>best_ppv: best_thr,best_ppv,best_sens=t,ppv,sens
if best_thr is None:
    best_thr=max(np.arange(0.01,0.51,0.01),key=lambda t: ((oof_prob>=t)&(y_all==1)).sum()/y_all.sum())
print(f"Chosen θ={best_thr:.2f}  (OOF Sens={best_sens:.3f}  PPV={best_ppv:.3f})")

# ---------- train full calibrated model ----------
full_model=lgb.LGBMClassifier(**lgb_params)
full_model.fit(X_all,y_all)
calibrated=CalibratedClassifierCV(full_model,method='isotonic',cv=5)
calibrated.fit(X_all,y_all)

def infer(df):
    X=df.drop(columns=["CH3","kind"]).fillna(train_df.median())
    prob=calibrated.predict_proba(X)[:,1]
    return (prob>=best_thr).astype(int),prob

# ---------- evaluate hold‑out + batch ----------
OUT="new_results/sens99_lgbm_run2"; os.makedirs(OUT,exist_ok=True)

def eval_file(pth,tag):
    df=load(pth); y=binarise(df["CH3"].astype(int)); pred,prob=infer(df); m,cm=metric_dict(y,pred,prob); save(tag,m,cm,OUT)
    print(f"{tag}: Sens={m['Sensitivity']:.3f} PPV={m['PPV']:.3f} Spec={m['Specificity']:.3f}")

eval_file("CH_exp1B_PreproccTotal_20_09_2024_10-20/Data_test_2022.csv","2022_Holdout")
BATCH="CH_exp1B_PreproccTotal_20_09_2024_10-20/Batch2 Data_test_2022.csv"
if os.path.exists(BATCH): eval_file(BATCH,"Batch_Final")
print("✅ LightGBM v2 done →",OUT)
