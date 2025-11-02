import json
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("OPTUNA XGBOOST TUNING")
print("Bayesian optimization to push past 95.33% test accuracy")
print("=" * 70)

train_df = pd.read_csv('train_processed.csv')
test_df = pd.read_csv('test_processed.csv')

X = train_df.drop(['ID', 'Overall_Experience'], axis=1)
y = train_df['Overall_Experience']
X_test = test_df.drop(['ID', 'Overall_Experience'], axis=1)
test_ids = test_df['ID']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training split: {len(X_train):,} train / {len(X_valid):,} valid")

def objective(trial):
    params = {
    'n_estimators': trial.suggest_int('n_estimators', 180, 360),
    'max_depth': trial.suggest_int('max_depth', 4, 8),
    'learning_rate': trial.suggest_float('learning_rate', 0.035, 0.1),
    'subsample': trial.suggest_float('subsample', 0.75, 0.95),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 0.95),
    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.3),
    'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 2.2),
    'min_child_weight': trial.suggest_int('min_child_weight', 2, 7),
    'gamma': trial.suggest_float('gamma', 0.0, 0.25),
    }

    model = xgb.XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_jobs=4,
        tree_method='hist',
        **params
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    best_iteration = getattr(model, 'best_iteration', params['n_estimators'])
    trial.set_user_attr('best_iteration', best_iteration)

    preds = model.predict(X_valid)
    acc = accuracy_score(y_valid, preds)
    return acc

study = optuna.create_study(direction='maximize')
print("Running Optuna study (10 trials)...")
study.optimize(objective, n_trials=10, show_progress_bar=False)

print("Best trial accuracy: {:.4f}".format(study.best_value))
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

best_params_full = study.best_params
best_n_estimators = best_params_full['n_estimators']
best_params = best_params_full.copy()
best_params.pop('n_estimators', None)
best_iteration = study.best_trial.user_attrs.get('best_iteration', best_n_estimators)
print(f"Best iteration (trees): {best_iteration}")

best_model = xgb.XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    n_jobs=4,
    tree_method='hist',
    n_estimators=best_iteration,
    **best_params
)

best_model.fit(X, y)

train_proba = best_model.predict_proba(X)[:, 1]
test_proba = best_model.predict_proba(X_test)[:, 1]

train_preds = (train_proba >= 0.5).astype(int)
test_preds = (test_proba >= 0.5).astype(int)

train_accuracy = (train_preds == y).mean()
print(f"Training accuracy at 0.5 threshold: {train_accuracy*100:.2f}%")

output_file = 'BulletBrain_Optuna_XGB.csv'
pd.DataFrame({'ID': test_ids, 'Overall_Experience': test_preds}).to_csv(output_file, index=False)
print(f"Saved: {output_file}")

proba_train_out = 'optuna_xgb_train_probs.csv'
proba_test_out = 'optuna_xgb_test_probs.csv'

pd.DataFrame({'ID': train_df['ID'], 'probability': train_proba, 'label': y}).to_csv(proba_train_out, index=False)
pd.DataFrame({'ID': test_ids, 'probability': test_proba}).to_csv(proba_test_out, index=False)

with open('optuna_xgb_best_params.json', 'w') as f:
    json.dump(best_params_full, f, indent=2)

print(f"Saved probability files: {proba_train_out}, {proba_test_out}")
print("Saved best params to optuna_xgb_best_params.json")

print("=" * 70)
print("Optuna tuning complete. Use threshold_calibration.py next.")
print("=" * 70)
