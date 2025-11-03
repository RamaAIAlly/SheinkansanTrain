"""
Quick Strategy to Beat 95.87% Accuracy
======================================
Focus on threshold optimization and quick ensemble
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

print("="*80)
print("QUICK STRATEGY: Beat 95.87% Accuracy")
print("="*80)

# Strategy 1: Optimize threshold on existing model
print("\n[1/3] Testing probability thresholds on existing model...")

train_probs_df = pd.read_csv('SheinkansenTrain_2Nov25/optuna_xgb_train_probs.csv')
train_data_survey = pd.read_csv('Surveydata_train.csv')
train_data_travel = pd.read_csv('Traveldata_train.csv')
train_data = pd.merge(train_data_survey, train_data_travel, on='ID', how='inner')

# Merge to get actual labels
merged = pd.merge(train_probs_df, train_data[['ID', 'Overall_Experience']], on='ID')
train_probs = merged['probability'].values
y_true = merged['Overall_Experience'].values

best_threshold = 0.5
best_acc = 0

print("Testing thresholds from 0.35 to 0.65...")
for threshold in np.arange(0.35, 0.66, 0.005):
    pred = (train_probs >= threshold).astype(int)
    acc = accuracy_score(y_true, pred)
    if acc > best_acc:
        best_acc = acc
        best_threshold = threshold
        print(f"  New best: threshold={threshold:.3f}, accuracy={acc:.6f}")

print(f"\nOptimal threshold: {best_threshold:.3f}")
print(f"Training accuracy: {best_acc*100:.4f}%")

# Apply optimal threshold to test predictions
print("\n[2/3] Generating predictions with optimal threshold...")

test_probs_df = pd.read_csv('SheinkansenTrain_2Nov25/optuna_xgb_test_probs.csv')
test_predictions = (test_probs_df['probability'] >= best_threshold).astype(int)

# Create submission v2
submission_v2 = pd.DataFrame({
    'ID': test_probs_df['ID'],
    'Overall_Experience': test_predictions
})
submission_v2 = submission_v2.sort_values('ID').reset_index(drop=True)

output_file = 'shinkansen_predictions_v2.csv'
submission_v2.to_csv(output_file, index=False)

n_satisfied = (test_predictions == 1).sum()
satisfaction_rate = n_satisfied / len(test_predictions) * 100

print(f"\n[3/3] Results:")
print(f"  Total predictions: {len(test_predictions):,}")
print(f"  Satisfied: {n_satisfied:,} ({satisfaction_rate:.2f}%)")
print(f"  Dissatisfied: {len(test_predictions)-n_satisfied:,} ({100-satisfaction_rate:.2f}%)")
print(f"  Output: {output_file}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"Original (threshold=0.5): 95.49% test accuracy")
print(f"Optimized (threshold={best_threshold:.3f}): {best_acc*100:.2f}% train accuracy")
print(f"Target to beat: 95.87%")
print("\nStrategy: Optimized decision threshold may close the gap!")
print("="*80)
