# src/train_model.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV

def load_processed(path_train, path_test):
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)
    X_train = train.drop(columns=['target'])
    y_train = train['target']
    X_test = test.drop(columns=['target'])
    y_test = test['target']
    return X_train, X_test, y_train, y_test

def train_baseline(X_train, y_train, n_estimators=200, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, save_predictions_path=None):
    y_pred = model.predict(X_test)
    out = pd.DataFrame(X_test.copy())
    out['y_true'] = y_test.values
    out['y_pred'] = y_pred
    if save_predictions_path:
        out.to_csv(save_predictions_path, index=False)
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    try:
        y_proba = model.predict_proba(X_test)[:,1]
        print("ROC AUC:", roc_auc_score(y_test, y_proba))
    except Exception as e:
        print("No predict_proba available:", e)
    return out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/processed/train.csv")
    parser.add_argument("--test_csv", default="data/processed/test.csv")
    parser.add_argument("--out_model", default="models/best_model.joblib")
    parser.add_argument("--out_preds", default="data/test_predictions.csv")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_processed(args.train_csv, args.test_csv)
    print("Train/test shapes:", X_train.shape, X_test.shape)
    model = train_baseline(X_train, y_train)
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    joblib.dump(model, args.out_model)
    print("Saved model to", args.out_model)
    evaluate_model(model, X_test, y_test, save_predictions_path=args.out_preds)
    print("Saved predictions to", args.out_preds)
