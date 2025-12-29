import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CHURN PREDICTION MODEL TRAINING")
print("="*60)

# Load preprocessed data
print("\nLoading preprocessed data...")
X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')

with open('data/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")
print(f"✓ Features: {len(feature_names)}")

# Function to evaluate model
def evaluate_model(name, model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics, y_pred, y_pred_proba

# Store results
results = []
models_dict = {}

print("\n" + "="*60)
print("MODEL 1: LOGISTIC REGRESSION (Baseline)")
print("="*60)

# Train Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
print("Training...")
lr_model.fit(X_train, y_train)
print("✓ Training complete")

# Evaluate
lr_metrics, lr_pred, lr_pred_proba = evaluate_model("Logistic Regression", lr_model, X_test, y_test)
results.append(lr_metrics)
models_dict['Logistic Regression'] = lr_model

print("\nPerformance Metrics:")
for key, value in lr_metrics.items():
    if key != 'Model':
        print(f"  {key}: {value:.4f}")

print("\n" + "="*60)
print("MODEL 2: RANDOM FOREST")
print("="*60)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)
print("Training...")
rf_model.fit(X_train, y_train)
print("✓ Training complete")

# Evaluate
rf_metrics, rf_pred, rf_pred_proba = evaluate_model("Random Forest", rf_model, X_test, y_test)
results.append(rf_metrics)
models_dict['Random Forest'] = rf_model

print("\nPerformance Metrics:")
for key, value in rf_metrics.items():
    if key != 'Model':
        print(f"  {key}: {value:.4f}")

# Compare models
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Find best model
best_model_name = results_df.loc[results_df['ROC AUC'].idxmax(), 'Model']
best_model = models_dict[best_model_name]
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   ROC AUC Score: {results_df['ROC AUC'].max():.4f}")

# Save best model
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"\n✓ Best model saved as 'best_model.pkl'")

# Save all models for comparison
with open('models/all_models.pkl', 'wb') as f:
    pickle.dump(models_dict, f)

# Get predictions from best model
if best_model_name == "Logistic Regression":
    best_pred = lr_pred
    best_pred_proba = lr_pred_proba
else:
    best_pred = rf_pred
    best_pred_proba = rf_pred_proba

# Create visualizations
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# 1. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('data/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix.png")

# 2. ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, best_pred_proba)
auc_score = roc_auc_score(y_test, best_pred_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {best_model_name}', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('data/roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_curve.png")

# 3. Feature Importance
plt.figure(figsize=(10, 8))
if best_model_name == "Logistic Regression":
    importance = np.abs(best_model.coef_[0])
else:
    importance = best_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values('Importance', ascending=False)

sns.barplot(data=feature_importance_df.head(10), x='Importance', y='Feature', palette='viridis')
plt.title(f'Top 10 Feature Importances - {best_model_name}', fontsize=14, fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('data/feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_importance.png")

# Save feature importance
feature_importance_df.to_csv('data/feature_importance.csv', index=False)
print("✓ Saved: feature_importance.csv")

# 4. Model Comparison Chart
plt.figure(figsize=(10, 6))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
x = np.arange(len(metrics_to_plot))
width = 0.35

for i, (_, row) in enumerate(results_df.iterrows()):
    values = [row[m] for m in metrics_to_plot]
    plt.bar(x + i*width, values, width, label=row['Model'])

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.xticks(x + width/2, metrics_to_plot, rotation=45, ha='right')
plt.legend()
plt.ylim([0, 1])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('data/model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison.png")

# Save results
results_df.to_csv('data/model_results.csv', index=False)
print("✓ Saved: model_results.csv")

print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, best_pred, target_names=['Retained', 'Churned']))

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print(f"\n• {cm[0][0]} customers correctly predicted as RETAINED")
print(f"• {cm[1][1]} customers correctly predicted as CHURNED")
print(f"• {cm[0][1]} FALSE POSITIVES (predicted churn but didn't)")
print(f"• {cm[1][0]} FALSE NEGATIVES (missed churn)")

# Business impact calculation
print("\n" + "="*60)
print("BUSINESS IMPACT ESTIMATION")
print("="*60)
avg_customer_ltv = 5000  # Assumed average customer lifetime value
retention_success_rate = 0.3  # Assumed 30% of interventions work

customers_saved = cm[1][1] * retention_success_rate
value_saved = customers_saved * avg_customer_ltv

print(f"\nAssumptions:")
print(f"• Average Customer LTV: ${avg_customer_ltv:,}")
print(f"• Retention Campaign Success Rate: {retention_success_rate*100:.0f}%")
print(f"\nPotential Impact:")
print(f"• Customers identified as at-risk: {cm[1][1] + cm[0][1]}")
print(f"• Potential customers saved: {customers_saved:.0f}")
print(f"• Estimated value saved: ${value_saved:,.0f}")

print("\n" + "="*60)
print("HOUR 2-3 COMPLETE! ✓")
print("="*60)
print("\nFiles created:")
print("• best_model.pkl - Your trained model")
print("• 4 visualization PNG files")
print("• model_results.csv - Performance metrics")
print("• feature_importance.csv - Feature rankings")
print("\nReady for Hour 4-5: Building the Streamlit App!")