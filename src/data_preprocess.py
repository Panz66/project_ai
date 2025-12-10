# src/data_preprocess.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_dataset(path):
    # support csv or excel
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.xls', '.xlsx']:
        # UCI original sometimes has header in row 1 (index 0), but first row is description.
        try:
            df = pd.read_excel(path, header=1)  # try header=1 (common for this dataset)
        except:
            df = pd.read_excel(path)
    else:
        # csv
        try:
            df = pd.read_csv(path, header=1)  # try header=1 first
        except:
            df = pd.read_csv(path)
    return df

def basic_cleaning(df):
    # rename target if different
    # common target names: 'default.payment.next.month' or 'default'
    possible_targets = ['default.payment.next.month', 'default', 'DEFAULT']
    target_col = None
    for t in possible_targets:
        if t in df.columns:
            target_col = t
            break
    if target_col is None:
        # try lowercase search
        for c in df.columns:
            if 'default' in c.lower() and ('month' in c.lower() or 'pay' in c.lower()):
                target_col = c
                break
    if target_col is None:
        raise KeyError("Tidak dapat menemukan kolom target (default). Periksa nama kolom dataset Anda.")
    # Drop ID if present
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    # ensure target named 'target'
    df = df.rename(columns={target_col: 'target'})
    # cast target to int (0/1)
    df['target'] = df['target'].astype(int)
    return df

def preprocess_and_split(df, test_size=0.2, random_state=42, stratify=True, scaler_out_path="models/scaler.joblib"):
    # basic cleaning
    df = basic_cleaning(df)
    # separate X,y
    y = df['target']
    X = df.drop(columns=['target'])

    # handle missing: fill numeric with median, categorical with mode
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    for c in num_cols:
        if X[c].isnull().any():
            X[c] = X[c].fillna(X[c].median())
    for c in obj_cols:
        if X[c].isnull().any():
            X[c] = X[c].fillna(X[c].mode().iloc[0])

    # simple one-hot for categorical (if any)
    if len(obj_cols) > 0:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

    # Split
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

    # scaling numeric columns (fit on train only)
    scaler = StandardScaler()
    num_cols_after = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train[num_cols_after] = scaler.fit_transform(X_train[num_cols_after])
    X_test[num_cols_after] = scaler.transform(X_test[num_cols_after])

    # save scaler
    os.makedirs(os.path.dirname(scaler_out_path), exist_ok=True)
    joblib.dump(scaler, scaler_out_path)

    # return as DataFrames
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess dataset and split train/test")
    parser.add_argument("--input", required=True, help="Path to raw dataset file (csv or xls/xlsx)")
    parser.add_argument("--out_dir", default="data/processed", help="Output directory for train/test CSVs")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    df = load_dataset(args.input)
    print("Loaded dataset shape:", df.shape)
    X_train, X_test, y_train, y_test = preprocess_and_split(df, test_size=args.test_size)

    os.makedirs(args.out_dir, exist_ok=True)
    train_df = X_train.copy()
    train_df['target'] = y_train.values
    test_df = X_test.copy()
    test_df['target'] = y_test.values
    train_df.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)
    print("Saved processed train/test to", args.out_dir)
