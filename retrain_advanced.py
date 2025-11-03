import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import json

print("="*80)
print("RETRAINING WITH ADVANCED FEATURES - FINAL ATTEMPT")
print("="*80)

train_df = pd.read_csv('train_processed.csv')
test_df = pd.read_csv('test_processed.csv')

X = train_df.drop(['ID', 'Overall_Experience'], axis=1)
y = train_df['Overall_Experience']
X_test = test_df.drop(['ID'], axis=1)
test_ids = test_df['ID']

print(f"Features: {X.shape[1]}")

with open('SheinkansenTrain_2Nov25/optuna_xgb_best_params.json', 'r') as f:
    best_params = json.load(f)

print("Training XGBoost...")
model = xgb.XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    tree_method='hist',
    **best_params
)

model.fit(X, y)

train_proba = model.predict_proba(X)[:, 1]
test_proba = model.predict_proba(X_test)[:, 1]

best_thresh = 0.5
best_acc = 0

for thresh in np.arange(0.45, 0.55, 0.005):
    pred = (train_proba >= thresh).astype(int)
    acc = accuracy_score(y, pred)
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

train_preds = (train_proba >= best_thresh).astype(int)
test_preds = (test_proba >= best_thresh).astype(int)

print(f"Threshold: {best_thresh:.3f}")
print(f"Training accuracy: {best_acc*100:.2f}%")

pd.DataFrame({
    'ID': test_ids, 
    'Overall_Experience': test_preds
}).to_csv('shinkansen_predictions_final.csv', index=False)

print(f"Satisfied: {test_preds.sum()} ({test_preds.sum()/len(test_preds)*100:.2f}%)")
print("Saved: shinkansen_predictions_final.csv")
