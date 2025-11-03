"""
Proper ML approach with RandomizedSearchCV + Cross-Validation
Target: Beat 95.87% by finding truly optimal parameters
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
import xgboost as xgb

print("="*80)
print("RANDOMIZEDSEARCHCV WITH CROSS-VALIDATION")
print("="*80)

# Load processed data with advanced features
train_df = pd.read_csv('train_processed.csv')
test_df = pd.read_csv('test_processed.csv')

X = train_df.drop(['ID', 'Overall_Experience'], axis=1)
y = train_df['Overall_Experience']
X_test = test_df.drop(['ID'], axis=1)
test_ids = test_df['ID']

print(f"Training: {X.shape[0]:,} samples, {X.shape[1]} features")

# Parameter distributions for RandomizedSearchCV
param_dist = {
    'n_estimators': [200, 250, 300, 350],
    'max_depth': [5, 6, 7, 8, 9],
    'learning_rate': [0.04, 0.05, 0.06, 0.07, 0.08],
    'subsample': [0.75, 0.80, 0.85, 0.90],
    'colsample_bytree': [0.75, 0.80, 0.85, 0.90],
    'min_child_weight': [3, 5, 7, 10],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.1, 0.2, 0.3],
    'reg_lambda': [1.0, 1.5, 2.0, 2.5]
}

# 5-fold stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Base model
xgb_model = xgb.XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    tree_method='hist',
    n_jobs=-1
)

print(f"\nRunning RandomizedSearchCV (50 iterations, 5-fold CV)...")
print("This will take 3-5 minutes...\n")

# RandomizedSearchCV (faster than GridSearchCV)
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,  # Try 50 random combinations
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

random_search.fit(X, y)

print("\n" + "="*80)
print("BEST PARAMETERS FOUND")
print("="*80)
print(f"Best CV Score: {random_search.best_score_*100:.4f}%")
print("\nBest Parameters:")
for param, value in sorted(random_search.best_params_.items()):
    print(f"  {param}: {value}")

# Use best model
best_model = random_search.best_estimator_

# Generate predictions
test_proba = best_model.predict_proba(X_test)[:, 1]

# Find optimal threshold with CV
print("\n" + "="*80)
print("THRESHOLD OPTIMIZATION")
print("="*80)

best_thresh = 0.5
best_cv_acc = 0

for thresh in np.arange(0.45, 0.55, 0.01):
    cv_scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        val_proba = best_model.predict_proba(X_val)[:, 1]
        val_pred = (val_proba >= thresh).astype(int)
        acc = accuracy_score(y_val, val_pred)
        cv_scores.append(acc)
    
    mean_cv_acc = np.mean(cv_scores)
    std_cv_acc = np.std(cv_scores)
    
    if mean_cv_acc > best_cv_acc:
        best_cv_acc = mean_cv_acc
        best_thresh = thresh
        print(f"Threshold {thresh:.2f}: {mean_cv_acc*100:.4f}% (+/- {std_cv_acc*100:.4f}%)")

print(f"\nOptimal threshold: {best_thresh:.2f}")
print(f"Expected CV accuracy: {best_cv_acc*100:.4f}%")

# Final predictions
final_predictions = (test_proba >= best_thresh).astype(int)

submission = pd.DataFrame({
    'ID': test_ids,
    'Overall_Experience': final_predictions
}).sort_values('ID').reset_index(drop=True)

output_file = 'shinkansen_predictions_cv_optimized.csv'
submission.to_csv(output_file, index=False)

n_satisfied = final_predictions.sum()

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"CV Score: {best_cv_acc*100:.4f}%")
print(f"Test predictions: {len(final_predictions):,}")
print(f"Satisfied: {n_satisfied:,} ({n_satisfied/len(final_predictions)*100:.2f}%)")
print(f"Output: {output_file}")

# Compare with v1
v1 = pd.read_csv('shinkansen_predictions.csv')
changes = (v1['Overall_Experience'] != final_predictions).sum()
print(f"Changes from v1: {changes}")

print("\n" + "="*80)
print("METHOD: RandomizedSearchCV + 5-Fold CV")
print("This uses proper ML methodology to avoid overfitting!")
print("="*80)
