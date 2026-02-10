import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)
import joblib
import os
import json

# Load and prepare the dataset
print("Loading dataset...")
ds = pd.read_csv('heart.csv')
print(f"Dataset shape: {ds.shape}")
print(f"Target distribution:\n{ds['target'].value_counts()}\n")

# Separate features and target
X = ds.drop('target', axis=1)
y = ds['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}\n")
streamlit run app.py
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store results
results = {}

def evaluate_model(model_name, y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics for a model"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary'),
        'Recall': recall_score(y_true, y_pred, average='binary'),
        'F1 Score': f1_score(y_true, y_pred, average='binary'),
        'MCC Score': matthews_corrcoef(y_true, y_pred)
    }
    
    # AUC score requires predicted probabilities
    if y_pred_proba is not None:
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            metrics['AUC Score'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            metrics['AUC Score'] = roc_auc_score(y_true, y_pred_proba)
    else:
        metrics['AUC Score'] = 'N/A'
    
    return metrics

def print_results(model_name, metrics):
    """Print formatted results for a model"""
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name:30s}: {value:.4f}")
        else:
            print(f"{metric_name:30s}: {value}")
    print(f"{'='*60}")

# ============================================================================
# 1. LOGISTIC REGRESSION
# ============================================================================
print("\n\nTraining Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)

lr_metrics = evaluate_model("Logistic Regression", y_test, lr_pred, lr_pred_proba)
results['Logistic Regression'] = lr_metrics
print_results("Logistic Regression", lr_metrics)

# ============================================================================
# 2. DECISION TREE CLASSIFIER
# ============================================================================
print("\n\nTraining Decision Tree Classifier...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_pred_proba = dt_model.predict_proba(X_test)

dt_metrics = evaluate_model("Decision Tree", y_test, dt_pred, dt_pred_proba)
results['Decision Tree'] = dt_metrics
print_results("Decision Tree Classifier", dt_metrics)

# ============================================================================
# 3. K-NEAREST NEIGHBORS CLASSIFIER
# ============================================================================
print("\n\nTraining K-Nearest Neighbors Classifier...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)
knn_pred_proba = knn_model.predict_proba(X_test_scaled)

knn_metrics = evaluate_model("K-Nearest Neighbors", y_test, knn_pred, knn_pred_proba)
results['K-Nearest Neighbors'] = knn_metrics
print_results("K-Nearest Neighbors Classifier", knn_metrics)

# ============================================================================
# 4. NAIVE BAYES CLASSIFIER (GAUSSIAN)
# ============================================================================
print("\n\nTraining Gaussian Naive Bayes Classifier...")
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
nb_pred = nb_model.predict(X_test_scaled)
nb_pred_proba = nb_model.predict_proba(X_test_scaled)

nb_metrics = evaluate_model("Gaussian Naive Bayes", y_test, nb_pred, nb_pred_proba)
results['Gaussian Naive Bayes'] = nb_metrics
print_results("Gaussian Naive Bayes Classifier", nb_metrics)

# ============================================================================
# 5. RANDOM FOREST CLASSIFIER
# ============================================================================
print("\n\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)

rf_metrics = evaluate_model("Random Forest", y_test, rf_pred, rf_pred_proba)
results['Random Forest'] = rf_metrics
print_results("Random Forest Classifier", rf_metrics)

# ============================================================================
# 6. XGBOOST CLASSIFIER
# ============================================================================
print("\n\nTraining XGBoost Classifier...")
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_pred_proba = xgb_model.predict_proba(X_test)

xgb_metrics = evaluate_model("XGBoost", y_test, xgb_pred, xgb_pred_proba)
results['XGBoost'] = xgb_metrics
print_results("XGBoost Classifier", xgb_metrics)

# ============================================================================
# SUMMARY OF ALL MODELS
# ============================================================================
print("\n\n" + "="*80)
print("SUMMARY: COMPARISON OF ALL MODELS")
print("="*80)

# Create a comparison dataframe
comparison_df = pd.DataFrame(results).T
print("\n", comparison_df.to_string())

# Find best model for each metric
print("\n\nBest Models by Metric:")
print("-" * 80)
for metric in ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score']:
    if metric in comparison_df.columns:
        if metric == 'AUC Score':
            # Filter out N/A values
            valid_scores = comparison_df[comparison_df[metric] != 'N/A'][metric]
            if len(valid_scores) > 0:
                best_model = valid_scores.astype(float).idxmax()
                best_score = valid_scores.astype(float).max()
                print(f"{metric:30s}: {best_model:30s} ({best_score:.4f})")
        else:
            best_model = comparison_df[metric].idxmax()
            best_score = comparison_df[metric].max()
            print(f"{metric:30s}: {best_model:30s} ({best_score:.4f})")

# Save results to JSON
print("\n\nSaving results to 'model_results.json'...")
# Convert any non-serializable values
results_serializable = {}
for model, metrics in results.items():
    results_serializable[model] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                   for k, v in metrics.items()}

with open('model_results.json', 'w') as f:
    json.dump(results_serializable, f, indent=4)

print("Results saved successfully!")

# ============================================================================
# SAVE TRAINED MODELS
# ============================================================================
print("\n\nSaving trained models to 'model/' directory...")

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save all models
joblib.dump(lr_model, 'model/logistic_regression.pkl')
joblib.dump(dt_model, 'model/decision_tree.pkl')
joblib.dump(knn_model, 'model/knn.pkl')
joblib.dump(nb_model, 'model/naive_bayes.pkl')
joblib.dump(rf_model, 'model/random_forest.pkl')
joblib.dump(xgb_model, 'model/xgboost.pkl')

# Save the scaler as well
joblib.dump(scaler, 'model/scaler.pkl')

print("✓ Logistic Regression model saved")
print("✓ Decision Tree model saved")
print("✓ K-Nearest Neighbors model saved")
print("✓ Naive Bayes model saved")
print("✓ Random Forest model saved")
print("✓ XGBoost model saved")
print("✓ Scaler saved")

print("\n" + "="*80)
print("Classification Analysis Complete!")
print("="*80)
