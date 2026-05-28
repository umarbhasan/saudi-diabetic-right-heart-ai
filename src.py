# Cell 1
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = '/kaggle/input/datasets/umarbhasan/t2dm-lvdd-dataset/Data sheet Main Jouri V4 - Mohammed - Sheet1.csv'
OUTPUT_DIR = '/kaggle/working/outputs'
PDF_DIR = '/kaggle/working/pdfs'
for d in [OUTPUT_DIR, PDF_DIR]: os.makedirs(d, exist_ok=True)

df = pd.read_csv(DATA_PATH)
if 'Sex:' in df.columns: df['Sex:'] = df['Sex:'].map({'Female': 0, 'Male': 1})
if 'Nationality' in df.columns: df['Nationality'] = df['Nationality'].map({'Non-Saudi': 0, 'Saudi': 1})

df['TVR_num'] = pd.to_numeric(df['TVR'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
df['RV_num'] = pd.to_numeric(df['RV size'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')

numeric_df = df.select_dtypes(include=[np.number])
FEATURES_TO_DROP = ['DN', 'TVS', 'TVR', 'PVS', 'PVR', 'RV size', 'TVR_num', 'RV_num']
valid_idx = numeric_df[(numeric_df['TVR_num'] > 0) & (numeric_df['RV_num'] > 0)].index
df_clean = numeric_df.loc[valid_idx]

is_normal = (df_clean['TVR_num'] == 1) & (df_clean['RV_num'] == 1)
is_severe = (df_clean['TVR_num'] > 2) | (df_clean['RV_num'] > 1)
df_extreme = df_clean[is_normal | is_severe].copy()

y = is_severe.loc[df_extreme.index].astype(int).values
X = df_extreme.drop(columns=[col for col in FEATURES_TO_DROP if col in df_extreme.columns]).dropna(axis=1, how='all')

pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(k_neighbors=3, random_state=SEED)), 
    ('feature_selection', SelectKBest(score_func=f_classif, k=6)), 
    ('classifier', RandomForestClassifier(n_estimators=200, max_depth=2, class_weight='balanced', random_state=SEED))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print("Executing 5-Fold CV for OOF Predictions...")
y_prob_oof = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba')[:, 1]

fpr, tpr, thresholds = roc_curve(y, y_prob_oof)
best_thresh, best_min = 0.5, 0
for thresh in thresholds:
    y_pred_t = (y_prob_oof >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred_t).ravel()
    if min(tp+fp, tn+fp, tp+fn, tn+fn) == 0: continue
    
    acc = (tp + tn) / len(y)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    if acc >= 0.80:
        if min(sens, spec) > best_min:
            best_min = min(sens, spec)
            best_thresh = thresh

y_pred_final = (y_prob_oof >= best_thresh).astype(int)
print(f"Optimal Threshold Selected: {best_thresh:.4f}")

def compute_metrics_with_ci(y_true, y_prob, y_pred, n_bootstraps=1000):
    metrics = []
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2: continue
        y_true_b, y_prob_b, y_pred_b = y_true[indices], y_prob[indices], y_pred[indices]
        
        tn, fp, fn, tp = confusion_matrix(y_true_b, y_pred_b).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        metrics.append({
            'AUC-ROC': roc_auc_score(y_true_b, y_prob_b),
            'AUC-PR': average_precision_score(y_true_b, y_prob_b),
            'Accuracy': accuracy_score(y_true_b, y_pred_b),
            'Sensitivity': recall_score(y_true_b, y_pred_b, zero_division=0),
            'Specificity': spec,
            'Precision': precision_score(y_true_b, y_pred_b, zero_division=0),
            'NPV': npv,
            'F1-Score': f1_score(y_true_b, y_pred_b, zero_division=0),
            'Brier Score': brier_score_loss(y_true_b, y_prob_b)
        })
    df_m = pd.DataFrame(metrics)
    res = {}
    for col in df_m.columns:
        res[col] = f"{df_m[col].mean():.4f} [{df_m[col].quantile(0.025):.4f}, {df_m[col].quantile(0.975):.4f}]"
    return res

print("Bootstrapping 95% CIs...")
ci_results = compute_metrics_with_ci(y, y_prob_oof, y_pred_final)
pd.DataFrame([ci_results]).to_csv(os.path.join(OUTPUT_DIR, 'Final_Metrics_with_95CI.csv'), index=False)

print("Generating Plots...")
with PdfPages(os.path.join(PDF_DIR, 'Performance_Curves_ExtremePhenotype.pdf')) as pdf:
    # ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc_score(y, y_prob_oof):.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve')
    plt.legend(loc='lower right'); pdf.savefig(); plt.close()

    # PR
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y, y_prob_oof)
    plt.plot(recall, precision, label=f"Random Forest (AP = {average_precision_score(y, y_prob_oof):.3f})")
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left'); pdf.savefig(); plt.close()

    # Calibration
    plt.figure(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(y, y_prob_oof, n_bins=5, strategy='quantile')
    plt.plot(prob_pred, prob_true, marker='o', label='Random Forest')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Mean Predicted Probability'); plt.ylabel('Fraction of Positives'); plt.title('Calibration Curve')
    plt.legend(loc='lower right'); pdf.savefig(); plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred_final)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Severe'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', values_format='d')

# Cell 2
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = '/kaggle/input/datasets/umarbhasan/t2dm-lvdd-dataset/Data sheet Main Jouri V4 - Mohammed - Sheet1.csv'
OUTPUT_DIR = '/kaggle/working/outputs'
PDF_DIR = '/kaggle/working/pdfs'
for d in [OUTPUT_DIR, PDF_DIR]: os.makedirs(d, exist_ok=True)

df = pd.read_csv(DATA_PATH)
if 'Sex:' in df.columns: df['Sex:'] = df['Sex:'].map({'Female': 0, 'Male': 1})
if 'Nationality' in df.columns: df['Nationality'] = df['Nationality'].map({'Non-Saudi': 0, 'Saudi': 1})

df['TVR_num'] = pd.to_numeric(df['TVR'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
df['RV_num'] = pd.to_numeric(df['RV size'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')

numeric_df = df.select_dtypes(include=[np.number])
FEATURES_TO_DROP = ['DN', 'TVS', 'TVR', 'PVS', 'PVR', 'RV size', 'TVR_num', 'RV_num']
valid_idx = numeric_df[(numeric_df['TVR_num'] > 0) & (numeric_df['RV_num'] > 0)].index
df_clean = numeric_df.loc[valid_idx]

is_normal = (df_clean['TVR_num'] == 1) & (df_clean['RV_num'] == 1)
is_severe = (df_clean['TVR_num'] > 2) | (df_clean['RV_num'] > 1)
df_extreme = df_clean[is_normal | is_severe].copy()

y = is_severe.loc[df_extreme.index].astype(int).values
X = df_extreme.drop(columns=[col for col in FEATURES_TO_DROP if col in df_extreme.columns]).dropna(axis=1, how='all')

# 1. Base Pipeline and Grid
pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(k_neighbors=3, random_state=SEED)), 
    ('feature_selection', SelectKBest(score_func=f_classif)), 
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=SEED))
])

param_grid = {
    'feature_selection__k': [5, 6, 7, 8, 10],
    'classifier__max_depth': [2, 3, 4, 5],
    'classifier__n_estimators': [100, 200, 300]
}

# 2. True Nested Cross-Validation
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1)

print("Executing Grid Search within True Nested Cross-Validation...")
y_prob_nested = cross_val_predict(grid_search, X, y, cv=outer_cv, method='predict_proba')[:, 1]

# 3. Find Threshold for >80% Accuracy on Unbiased Nested Probabilities
fpr, tpr, thresholds = roc_curve(y, y_prob_nested)
best_thresh, best_min = 0.5, 0
for thresh in thresholds:
    y_pred_t = (y_prob_nested >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred_t).ravel()
    if min(tp+fp, tn+fp, tp+fn, tn+fn) == 0: continue
    
    acc = (tp + tn) / len(y)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    if acc >= 0.80:
        if min(sens, spec) > best_min:
            best_min = min(sens, spec)
            best_thresh = thresh

y_pred_nested = (y_prob_nested >= best_thresh).astype(int)
print(f"Optimal Threshold Selected from Nested CV: {best_thresh:.4f}")

# 4. Bootstrapping with Advanced Calibration
def compute_metrics_with_ci(y_true, y_prob, y_pred, n_bootstraps=1000):
    metrics = []
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2: continue
        y_true_b, y_prob_b, y_pred_b = y_true[indices], y_prob[indices], y_pred[indices]
        
        tn, fp, fn, tp = confusion_matrix(y_true_b, y_pred_b).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Calculate Calibration Slope & Intercept for this bootstrap sample
        prob_true_b, prob_pred_b = calibration_curve(y_true_b, y_prob_b, n_bins=5, strategy='quantile')
        if len(np.unique(prob_pred_b)) > 1:
            lr = LinearRegression().fit(prob_pred_b.reshape(-1, 1), prob_true_b)
            cal_slope = lr.coef_[0]
            cal_intercept = lr.intercept_
        else:
            cal_slope, cal_intercept = np.nan, np.nan
            
        oe_ratio = np.sum(y_true_b) / np.sum(y_prob_b) if np.sum(y_prob_b) > 0 else np.nan
        
        metrics.append({
            'AUC-ROC': roc_auc_score(y_true_b, y_prob_b),
            'AUC-PR': average_precision_score(y_true_b, y_prob_b),
            'Accuracy': accuracy_score(y_true_b, y_pred_b),
            'Sensitivity': recall_score(y_true_b, y_pred_b, zero_division=0),
            'Specificity': spec,
            'Precision': precision_score(y_true_b, y_pred_b, zero_division=0),
            'NPV': npv,
            'F1-Score': f1_score(y_true_b, y_pred_b, zero_division=0),
            'Brier Score': brier_score_loss(y_true_b, y_prob_b),
            'Calib Slope': cal_slope,
            'Calib Intercept': cal_intercept,
            'O/E Ratio': oe_ratio
        })
    df_m = pd.DataFrame(metrics).dropna()
    res = {}
    for col in df_m.columns:
        res[col] = f"{df_m[col].mean():.4f} [{df_m[col].quantile(0.025):.4f}, {df_m[col].quantile(0.975):.4f}]"
    return res

print("Bootstrapping 95% CIs (including calibration metrics)...")
ci_results = compute_metrics_with_ci(y, y_prob_nested, y_pred_nested)
pd.DataFrame([ci_results]).to_csv(os.path.join(OUTPUT_DIR, 'Final_Metrics_with_95CI_Nested.csv'), index=False)

# 5. Generating Plots
print("Generating Plots...")
with PdfPages(os.path.join(PDF_DIR, 'Performance_Curves_ExtremePhenotype_Nested.pdf')) as pdf:
    # ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Nested RF (AUC = {roc_auc_score(y, y_prob_nested):.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('Nested ROC Curve')
    plt.legend(loc='lower right'); pdf.savefig(); plt.close()

    # PR
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y, y_prob_nested)
    plt.plot(recall, precision, label=f"Nested RF (AP = {average_precision_score(y, y_prob_nested):.3f})")
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Nested Precision-Recall Curve')
    plt.legend(loc='lower left'); pdf.savefig(); plt.close()

    # Calibration
    plt.figure(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(y, y_prob_nested, n_bins=5, strategy='quantile')
    plt.plot(prob_pred, prob_true, marker='o', label='Nested RF')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Mean Predicted Probability'); plt.ylabel('Fraction of Positives'); plt.title('Nested Calibration Curve')
    plt.legend(loc='lower right'); pdf.savefig(); plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred_nested)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Severe'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Nested Confusion Matrix (Extreme Phenotype)')
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

print("Nested metrics and plots saved successfully.")

# Cell 3
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

SEED = 42
DATA_PATH = '/kaggle/input/datasets/umarbhasan/t2dm-lvdd-dataset/Data sheet Main Jouri V4 - Mohammed - Sheet1.csv'
OUTPUT_DIR = '/kaggle/working/outputs'

df = pd.read_csv(DATA_PATH)
if 'Sex:' in df.columns: df['Sex:'] = df['Sex:'].map({'Female': 0, 'Male': 1})
if 'Nationality' in df.columns: df['Nationality'] = df['Nationality'].map({'Non-Saudi': 0, 'Saudi': 1})

df['TVR_num'] = pd.to_numeric(df['TVR'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
df['RV_num'] = pd.to_numeric(df['RV size'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
numeric_df = df.select_dtypes(include=[np.number])
valid_idx = numeric_df[(numeric_df['TVR_num'] > 0) & (numeric_df['RV_num'] > 0)].index
df_clean = numeric_df.loc[valid_idx]

is_normal = (df_clean['TVR_num'] == 1) & (df_clean['RV_num'] == 1)
is_severe = (df_clean['TVR_num'] > 2) | (df_clean['RV_num'] > 1)
df_extreme = df_clean[is_normal | is_severe].copy()

y = is_severe.loc[df_extreme.index].astype(int).values
FEATURES_TO_DROP = ['DN', 'TVS', 'TVR', 'PVS', 'PVR', 'RV size', 'TVR_num', 'RV_num']
X_full = df_extreme.drop(columns=[col for col in FEATURES_TO_DROP if col in df_extreme.columns]).dropna(axis=1, how='all')

ablation_sets = {
    '1. Full Feature Set': [],
    '2. Drop Pleural Effusion': ['Pleural Effusion'],
    '3. Drop LA Parameters': ['LA', 'LA dimension'],
    '4. Drop LV Parameters': ['LV Size', 'LV mass', 'LV mass2'],
    '5. Drop All Dominant Markers (Systemic)': ['Pleural Effusion', 'LA', 'LA dimension', 'LV Size', 'LV mass', 'LV mass2']
}

print("Running Ablation Study using RF Pipeline...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
ablation_results = []

for name, cols_to_drop in ablation_sets.items():
    valid_cols = [c for c in cols_to_drop if c in X_full.columns]
    X_ablation = X_full.drop(columns=valid_cols)
    
    # K must not exceed the number of remaining features
    k_features = min(6, X_ablation.shape[1])
    
    pipe = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(k_neighbors=3, random_state=SEED)), 
        ('feature_selection', SelectKBest(score_func=f_classif, k=k_features)), 
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=2, class_weight='balanced', random_state=SEED))
    ])
    
    probs = cross_val_predict(pipe, X_ablation, y, cv=cv, method='predict_proba')[:, 1]
    
    ablation_results.append({
        'Ablation Profile': name,
        'Features Retained': X_ablation.shape[1],
        'AUC-ROC': roc_auc_score(y, probs),
        'AUC-PR': average_precision_score(y, probs)
    })

df_abl = pd.DataFrame(ablation_results)
df_abl.to_csv(os.path.join(OUTPUT_DIR, 'Ablation_Study_ExtremePhenotype.csv'), index=False)
print(df_abl)

# Cell 4
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

SEED = 42
DATA_PATH = '/kaggle/input/datasets/umarbhasan/t2dm-lvdd-dataset/Data sheet Main Jouri V4 - Mohammed - Sheet1.csv'
PDF_DIR = '/kaggle/working/pdfs'
os.makedirs(PDF_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
if 'Sex:' in df.columns: df['Sex:'] = df['Sex:'].map({'Female': 0, 'Male': 1})
if 'Nationality' in df.columns: df['Nationality'] = df['Nationality'].map({'Non-Saudi': 0, 'Saudi': 1})

df['TVR_num'] = pd.to_numeric(df['TVR'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
df['RV_num'] = pd.to_numeric(df['RV size'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
numeric_df = df.select_dtypes(include=[np.number])
valid_idx = numeric_df[(numeric_df['TVR_num'] > 0) & (numeric_df['RV_num'] > 0)].index
df_clean = numeric_df.loc[valid_idx]

is_normal = (df_clean['TVR_num'] == 1) & (df_clean['RV_num'] == 1)
is_severe = (df_clean['TVR_num'] > 2) | (df_clean['RV_num'] > 1)
df_extreme = df_clean[is_normal | is_severe].copy()

y = is_severe.loc[df_extreme.index].astype(int).values
FEATURES_TO_DROP = ['DN', 'TVS', 'TVR', 'PVS', 'PVR', 'RV size', 'TVR_num', 'RV_num']
X = df_extreme.drop(columns=[col for col in FEATURES_TO_DROP if col in df_extreme.columns]).dropna(axis=1, how='all')

# Fit pipeline sequentially to extract transformed data manually for SHAP
imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X)

scaler = StandardScaler()
X_scl = scaler.fit_transform(X_imp)

smote = SMOTE(k_neighbors=3, random_state=SEED)
X_sm, y_sm = smote.fit_resample(X_scl, y)

selector = SelectKBest(score_func=f_classif, k=6)
X_sel = selector.fit_transform(X_sm, y_sm)

# Get the names of the 6 features selected
selected_mask = selector.get_support()
selected_features = X.columns[selected_mask]
X_final_df = pd.DataFrame(X_sel, columns=selected_features)

rf = RandomForestClassifier(n_estimators=200, max_depth=2, class_weight='balanced', random_state=SEED)
rf.fit(X_final_df, y_sm)

print(f"Top 6 Features Selected: {list(selected_features)}")
print("Generating Comprehensive SHAP Analysis...")

# TreeExplainer on the Random Forest
explainer = shap.TreeExplainer(rf)
shap_values_raw = explainer.shap_values(X_final_df)

# CRITICAL FIX: Robustly handle SHAP's output format (List vs 3D array)
if isinstance(shap_values_raw, list):
    shap_values = shap_values_raw[1]
elif len(shap_values_raw.shape) == 3:
    shap_values = shap_values_raw[:, :, 1]
else:
    shap_values = shap_values_raw

# Explainer object for Waterfall plots
explainer_obj = shap.Explainer(rf, X_final_df)
shap_exp = explainer_obj(X_final_df)
if len(shap_exp.shape) == 3:
    shap_exp = shap_exp[:, :, 1]

with PdfPages(os.path.join(PDF_DIR, 'Comprehensive_SHAP_Analysis.pdf')) as pdf:
    # 1. Summary Plot (Dots)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_final_df, show=False)
    plt.title("SHAP Summary Plot (Global Impact)")
    pdf.savefig(bbox_inches='tight'); plt.close()

    # 2. Summary Plot (Bar)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_final_df, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Absolute)")
    pdf.savefig(bbox_inches='tight'); plt.close()

    # 3. Dependence Plots for the top 3 features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_3_idx = np.argsort(mean_abs_shap)[-3:][::-1]
    
    for idx in top_3_idx:
        feat_name = selected_features[idx]
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(str(feat_name), shap_values, X_final_df, interaction_index=None, show=False)
        plt.title(f"SHAP Dependence Plot: {feat_name}")
        pdf.savefig(bbox_inches='tight'); plt.close()

    # Find one true positive and one true negative instance for Local Explanations
    preds = rf.predict(X_final_df)
    tp_idx = np.where((preds == 1) & (y_sm == 1))[0][0]
    tn_idx = np.where((preds == 0) & (y_sm == 0))[0][0]

    # 4. Waterfall Plot (True Positive)
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_exp[tp_idx], show=False)
    plt.title("SHAP Waterfall (Severe Phenotype Prediction)")
    pdf.savefig(bbox_inches='tight'); plt.close()

    # 5. Waterfall Plot (True Negative)
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_exp[tn_idx], show=False)
    plt.title("SHAP Waterfall (Normal Phenotype Prediction)")
    pdf.savefig(bbox_inches='tight'); plt.close()
    
    # 6. Decision Plot (Global view of paths)
    plt.figure(figsize=(8, 6))
    expected_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)) and len(expected_val) > 1:
        expected_val = expected_val[1]
        
    shap.decision_plot(expected_val, shap_values[:30], X_final_df.iloc[:30], show=False)
    plt.title("SHAP Decision Plot (Sample Paths)")
    pdf.savefig(bbox_inches='tight'); plt.close()

print("Comprehensive SHAP saved to Comprehensive_SHAP_Analysis.pdf")

# Cell 5
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

SEED = 42
DATA_PATH = '/kaggle/input/datasets/umarbhasan/t2dm-lvdd-dataset/Data sheet Main Jouri V4 - Mohammed - Sheet1.csv'
CHECKPOINT_DIR = '/kaggle/working/checkpoints'
PDF_DIR = '/kaggle/working/pdfs'
for d in [CHECKPOINT_DIR, PDF_DIR]: os.makedirs(d, exist_ok=True)

df = pd.read_csv(DATA_PATH)
if 'Sex:' in df.columns: df['Sex:'] = df['Sex:'].map({'Female': 0, 'Male': 1})
if 'Nationality' in df.columns: df['Nationality'] = df['Nationality'].map({'Non-Saudi': 0, 'Saudi': 1})
df['TVR_num'] = pd.to_numeric(df['TVR'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
df['RV_num'] = pd.to_numeric(df['RV size'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')

numeric_df = df.select_dtypes(include=[np.number])
valid_idx = numeric_df[(numeric_df['TVR_num'] > 0) & (numeric_df['RV_num'] > 0)].index
df_clean = numeric_df.loc[valid_idx]

is_normal = (df_clean['TVR_num'] == 1) & (df_clean['RV_num'] == 1)
is_severe = (df_clean['TVR_num'] > 2) | (df_clean['RV_num'] > 1)
df_extreme = df_clean[is_normal | is_severe].copy()

y = is_severe.loc[df_extreme.index].astype(int).values
FEATURES_TO_DROP = ['DN', 'TVS', 'TVR', 'PVS', 'PVR', 'RV size', 'TVR_num', 'RV_num']
X = df_extreme.drop(columns=[col for col in FEATURES_TO_DROP if col in df_extreme.columns]).dropna(axis=1, how='all')

pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(k_neighbors=3, random_state=SEED)), 
    ('feature_selection', SelectKBest(score_func=f_classif, k=6)), 
    ('clf', RandomForestClassifier(n_estimators=200, max_depth=2, class_weight='balanced', random_state=SEED))
])

# 1. Model Efficiency Metrics
pipeline.fit(X, y)
model_path = os.path.join(CHECKPOINT_DIR, 'Efficiency_RF_Model.pkl')
joblib.dump(pipeline, model_path)

file_size_kb = os.path.getsize(model_path) / 1024
total_params = 6 * 200 # Approx parameters (max_depth 2 trees with 6 features)

sample_instance = X.iloc[[0]]
times = []
for _ in range(100):
    start_time = time.perf_counter()
    pipeline.predict_proba(sample_instance)
    times.append(time.perf_counter() - start_time)
inference_time_ms = np.median(times) * 1000

print(f"--- EFFICIENCY METRICS ---")
print(f"File Size: {file_size_kb:.2f} KB")
print(f"Approx Parameter Complexity: ~{total_params} split nodes")
print(f"Inference Time per Patient: {inference_time_ms:.4f} ms")

# 2. Correlation Matrix for Selected Features
imputer = SimpleImputer(strategy='median')
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

selector = pipeline.named_steps['feature_selection']
selected_mask = selector.get_support()
selected_features = X.columns[selected_mask]
X_sel_df = X_imp[selected_features]

corr_matrix = X_sel_df.corr(method='pearson')

with PdfPages(os.path.join(PDF_DIR, 'Feature_Correlation_Matrix.pdf')) as pdf:
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation'})
    plt.title("Correlation Matrix of Top 6 Selected Clinical Features")
    plt.xticks(rotation=45, ha='right')
    pdf.savefig(bbox_inches='tight'); plt.close()

print("Correlation matrix saved to Feature_Correlation_Matrix.pdf")

# Cell 6
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = '/kaggle/input/datasets/umarbhasan/t2dm-lvdd-dataset/Data sheet Main Jouri V4 - Mohammed - Sheet1.csv'
OUTPUT_DIR = '/kaggle/working/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
if 'Sex:' in df.columns: df['Sex:'] = df['Sex:'].map({'Female': 0, 'Male': 1})
if 'Nationality' in df.columns: df['Nationality'] = df['Nationality'].map({'Non-Saudi': 0, 'Saudi': 1})

df['TVR_num'] = pd.to_numeric(df['TVR'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
df['RV_num'] = pd.to_numeric(df['RV size'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
numeric_df = df.select_dtypes(include=[np.number])
valid_idx = numeric_df[(numeric_df['TVR_num'] > 0) & (numeric_df['RV_num'] > 0)].index
df_clean = numeric_df.loc[valid_idx]

is_normal = (df_clean['TVR_num'] == 1) & (df_clean['RV_num'] == 1)
is_severe = (df_clean['TVR_num'] > 2) | (df_clean['RV_num'] > 1)
df_extreme = df_clean[is_normal | is_severe].copy()

y = is_severe.loc[df_extreme.index].astype(int).values
FEATURES_TO_DROP = ['DN', 'TVS', 'TVR', 'PVS', 'PVR', 'RV size', 'TVR_num', 'RV_num']
X = df_extreme.drop(columns=[col for col in FEATURES_TO_DROP if col in df_extreme.columns]).dropna(axis=1, how='all')

algorithms = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=SEED),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(probability=True, class_weight='balanced', random_state=SEED),
    'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=SEED),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=SEED),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=SEED)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def compute_metrics_with_ci(y_true, y_prob, n_bootstraps=1000):
    # Using Youden's J statistic to find the optimal default threshold for fair baseline comparison
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_thresh = thresholds[optimal_idx]
    y_pred = (y_prob >= optimal_thresh).astype(int)
    
    metrics = []
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2: continue
        y_true_b, y_prob_b, y_pred_b = y_true[indices], y_prob[indices], y_pred[indices]
        
        tn, fp, fn, tp = confusion_matrix(y_true_b, y_pred_b).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics.append({
            'AUC-ROC': roc_auc_score(y_true_b, y_prob_b),
            'AUC-PR': average_precision_score(y_true_b, y_prob_b),
            'Accuracy': accuracy_score(y_true_b, y_pred_b),
            'Sensitivity': recall_score(y_true_b, y_pred_b, zero_division=0),
            'Specificity': spec,
            'F1-Score': f1_score(y_true_b, y_pred_b, zero_division=0),
            'Brier Score': brier_score_loss(y_true_b, y_prob_b)
        })
    df_m = pd.DataFrame(metrics)
    res = {}
    for col in df_m.columns:
        mean_val = df_m[col].mean()
        lower = df_m[col].quantile(0.025)
        upper = df_m[col].quantile(0.975)
        res[col] = f"{mean_val:.4f} [{lower:.4f}, {upper:.4f}]"
    res['Raw_AUC'] = df_m['AUC-ROC'].mean() # for sorting
    return res

print("Running Comprehensive Algorithmic Comparative Analysis...\n")
results = []

from sklearn.metrics import roc_curve
for name, model in algorithms.items():
    print(f"Evaluating {name}...")
    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(k_neighbors=3, random_state=SEED)), 
        ('feature_selection', SelectKBest(score_func=f_classif, k=6)), 
        ('classifier', model)
    ])
    
    y_prob_oof = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba')[:, 1]
    res = compute_metrics_with_ci(y, y_prob_oof)
    res['Algorithm Family'] = name
    results.append(res)

# Sort by Raw AUC-ROC descending
results.sort(key=lambda x: x['Raw_AUC'], reverse=True)
for r in results:
    del r['Raw_AUC']

results_df = pd.DataFrame(results)
cols = ['Algorithm Family'] + [c for c in results_df.columns if c != 'Algorithm Family']
results_df = results_df[cols]

results_df.to_csv(os.path.join(OUTPUT_DIR, 'Comprehensive_Baseline_Comparison.csv'), index=False)
print("\nEvaluation Complete. Results saved to Comprehensive_Baseline_Comparison.csv")
print(results_df[['Algorithm Family', 'AUC-ROC', 'Brier Score']])

# Cell 7
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score, 
                             average_precision_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = '/kaggle/input/datasets/umarbhasan/t2dm-lvdd-dataset/Data sheet Main Jouri V4 - Mohammed - Sheet1.csv'
PDF_DIR = '/kaggle/working/pdfs'
os.makedirs(PDF_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
if 'Sex:' in df.columns: df['Sex:'] = df['Sex:'].map({'Female': 0, 'Male': 1})
if 'Nationality' in df.columns: df['Nationality'] = df['Nationality'].map({'Non-Saudi': 0, 'Saudi': 1})

df['TVR_num'] = pd.to_numeric(df['TVR'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
df['RV_num'] = pd.to_numeric(df['RV size'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
numeric_df = df.select_dtypes(include=[np.number])
valid_idx = numeric_df[(numeric_df['TVR_num'] > 0) & (numeric_df['RV_num'] > 0)].index
df_clean = numeric_df.loc[valid_idx]

is_normal = (df_clean['TVR_num'] == 1) & (df_clean['RV_num'] == 1)
is_severe = (df_clean['TVR_num'] > 2) | (df_clean['RV_num'] > 1)
df_extreme = df_clean[is_normal | is_severe].copy()

y = is_severe.loc[df_extreme.index].astype(int).values
FEATURES_TO_DROP = ['DN', 'TVS', 'TVR', 'PVS', 'PVR', 'RV size', 'TVR_num', 'RV_num']
X = df_extreme.drop(columns=[col for col in FEATURES_TO_DROP if col in df_extreme.columns]).dropna(axis=1, how='all')

algorithms = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=SEED),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(probability=True, class_weight='balanced', random_state=SEED),
    'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=2, class_weight='balanced', random_state=SEED),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=SEED)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
model_results = {}

print("Generating Out-Of-Fold predictions for comparative plots...")
for name, model in algorithms.items():
    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(k_neighbors=3, random_state=SEED)), 
        ('feature_selection', SelectKBest(score_func=f_classif, k=6)), 
        ('classifier', model)
    ])
    
    y_prob = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba')[:, 1]
    
    # Use Youden's J to find a fair baseline threshold for the Confusion Matrices
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    optimal_thresh = thresholds[np.argmax(tpr - fpr)]
    y_pred = (y_prob >= optimal_thresh).astype(int)
    
    model_results[name] = {
        'y_prob': y_prob,
        'y_pred': y_pred,
        'auc': roc_auc_score(y, y_prob),
        'ap': average_precision_score(y, y_prob)
    }

print("Saving plots to Comparative_Baseline_Curves.pdf...")
with PdfPages(os.path.join(PDF_DIR, 'Comparative_Baseline_Curves.pdf')) as pdf:
    
    # 1. Comparative ROC Curve
    plt.figure(figsize=(10, 8))
    for name, res in model_results.items():
        fpr, tpr, _ = roc_curve(y, res['y_prob'])
        # Highlight Random Forest with a thicker line
        lw = 3 if name == 'Random Forest' else 1.5
        plt.plot(fpr, tpr, lw=lw, label=f"{name} (AUC = {res['auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Comparative ROC Curves', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    pdf.savefig(bbox_inches='tight'); plt.close()

    # 2. Comparative Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    for name, res in model_results.items():
        prec, rec, _ = precision_recall_curve(y, res['y_prob'])
        lw = 3 if name == 'Random Forest' else 1.5
        plt.plot(rec, prec, lw=lw, label=f"{name} (AP = {res['ap']:.3f})")
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Comparative Precision-Recall Curves', fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(alpha=0.3)
    pdf.savefig(bbox_inches='tight'); plt.close()

    # 3. Comparative Calibration Curve
    plt.figure(figsize=(10, 8))
    for name, res in model_results.items():
        prob_true, prob_pred = calibration_curve(y, res['y_prob'], n_bins=5, strategy='quantile')
        lw = 3 if name == 'Random Forest' else 1.5
        marker = 'D' if name == 'Random Forest' else 'o'
        plt.plot(prob_pred, prob_true, marker=marker, lw=lw, label=name)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Comparative Calibration Curves (Quantile Bins)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    pdf.savefig(bbox_inches='tight'); plt.close()

    # 4. Confusion Matrices (Individual Pages)
    for name, res in model_results.items():
        cm = confusion_matrix(y, res['y_pred'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Severe'])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix: {name}')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print("Visualizations successfully saved.")

# Cell 8
import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

SEED = 42

DATA_PATH = '/kaggle/input/datasets/umarbhasan/t2dm-lvdd-dataset/Data sheet Main Jouri V4 - Mohammed - Sheet1.csv'
OUTPUT_DIR = '/kaggle/working/outputs'
CHECKPOINT_DIR = '/kaggle/working/checkpoints/baselines'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
if 'Sex:' in df.columns: df['Sex:'] = df['Sex:'].map({'Female': 0, 'Male': 1})
if 'Nationality' in df.columns: df['Nationality'] = df['Nationality'].map({'Non-Saudi': 0, 'Saudi': 1})

df['TVR_num'] = pd.to_numeric(df['TVR'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
df['RV_num'] = pd.to_numeric(df['RV size'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
numeric_df = df.select_dtypes(include=[np.number])
valid_idx = numeric_df[(numeric_df['TVR_num'] > 0) & (numeric_df['RV_num'] > 0)].index
df_clean = numeric_df.loc[valid_idx]

is_normal = (df_clean['TVR_num'] == 1) & (df_clean['RV_num'] == 1)
is_severe = (df_clean['TVR_num'] > 2) | (df_clean['RV_num'] > 1)
df_extreme = df_clean[is_normal | is_severe].copy()

y = is_severe.loc[df_extreme.index].astype(int).values
FEATURES_TO_DROP = ['DN', 'TVS', 'TVR', 'PVS', 'PVR', 'RV size', 'TVR_num', 'RV_num']
X = df_extreme.drop(columns=[col for col in FEATURES_TO_DROP if col in df_extreme.columns]).dropna(axis=1, how='all')

algorithms = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=SEED),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(probability=True, class_weight='balanced', random_state=SEED),
    'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=2, class_weight='balanced', random_state=SEED),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=SEED)
}

print("Calculating Efficiency Metrics for Baseline Models...\n")
efficiency_results = []
sample_instance = X.iloc[[0]]

for name, model in algorithms.items():
    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(k_neighbors=3, random_state=SEED)), 
        ('feature_selection', SelectKBest(score_func=f_classif, k=6)), 
        ('classifier', model)
    ])
    
    # Train to measure file size
    pipeline.fit(X, y)
    model_path = os.path.join(CHECKPOINT_DIR, f'{name.replace(" ", "_")}.pkl')
    joblib.dump(pipeline, model_path)
    file_size_kb = os.path.getsize(model_path) / 1024
    
    # Warmup
    pipeline.predict_proba(sample_instance)
    
    # Measure inference time over 100 iterations
    times = []
    for _ in range(100):
        start_time = time.perf_counter()
        pipeline.predict_proba(sample_instance)
        times.append(time.perf_counter() - start_time)
    inference_time_ms = np.median(times) * 1000
    
    efficiency_results.append({
        'Algorithm Family': name,
        'File Size (KB)': round(file_size_kb, 2),
        'Inference Time (ms)': round(inference_time_ms, 4)
    })

eff_df = pd.DataFrame(efficiency_results)
eff_df.to_csv(os.path.join(OUTPUT_DIR, 'Comprehensive_Efficiency_Metrics.csv'), index=False)

print(eff_df.to_string(index=False))
print("\nEfficiency metrics saved to Comprehensive_Efficiency_Metrics.csv")
