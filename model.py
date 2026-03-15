

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# 1. Load and explore data
# ---------------------------
df = pd.read_csv('train.csv')
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['Heart Disease'].value_counts()}")

# Map target to binary (0 = Absence, 1 = Presence)
df['Heart Disease'] = df['Heart Disease'].map({'Absence': 0, 'Presence': 1})

# Check for missing values
print(f"Missing values:\n{df.isnull().sum()}")

# Separate features and target
X = df.drop(['id', 'Heart Disease'], axis=1)
y = df['Heart Disease']
feature_names = X.columns.tolist()

# ---------------------------
# 2. Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ---------------------------
# 3. Feature scaling
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 4. Handle class imbalance (if needed)
# ---------------------------
print(f"Class distribution in training set:\n{y_train.value_counts()}")
if y_train.value_counts()[1] / y_train.value_counts()[0] < 0.7:
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
    print(f"New training set size after SMOTE: {X_train_scaled.shape}")
else:
    print("Classes are sufficiently balanced; skipping SMOTE.")

# ---------------------------
# 5. Model definitions
# ---------------------------
# Hyperparameters chosen based on typical literature and initial experiments:
# - RandomForest: 100 trees, max depth 10 to prevent overfitting, balanced class weight.
# - LogisticRegression: L2 regularization, C=1.0, solver='liblinear' for small dataset, balanced class weight.
# - XGBoost: 100 estimators, learning rate 0.1, max depth 6, scale_pos_weight to handle imbalance.
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    ),
    'Logistic Regression': LogisticRegression(
        C=1.0,
        solver='liblinear',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train)  # balance positive class
    )
}

# ---------------------------
# 6. Train and evaluate models
# ---------------------------
results = {}
conf_matrices = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    conf_matrices[name] = cm

    print(f"{name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

# ---------------------------
# 7. Save models and scaler
# ---------------------------
joblib.dump(scaler, 'scaler.pkl')
for name, model in models.items():
    # Clean filename: replace spaces with underscores, lowercase
    filename = name.replace(' ', '_').lower() + '.pkl'
    joblib.dump(model, filename)

# Save evaluation results for frontend
results_df = pd.DataFrame(results).T
results_df.to_csv('model_results.csv')
print("\nModel results saved to model_results.csv")

# ---------------------------
# 8. Generate SHAP explanations for the best model (XGBoost as example)
# ---------------------------
print("\nGenerating SHAP summary plot for XGBoost...")
X_train_sample = X_train_scaled[:100]  # using 100 samples
explainer = shap.TreeExplainer(models['XGBoost'])
shap_values = explainer.shap_values(X_train_sample)

# Plot summary
shap.summary_plot(shap_values, X_train_sample, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("SHAP plot saved as shap_summary.png")

# ---------------------------
# 9. Plot confusion matrices and save
# ---------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, cm) in zip(axes, conf_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Absence', 'Presence'],
                yticklabels=['Absence', 'Presence'])
    ax.set_title(f'{name} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.close()
print("Confusion matrices saved as confusion_matrices.png")

print("\nTraining pipeline completed successfully!")
