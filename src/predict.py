# src/predict.py
import joblib
import pandas as pd
import os

def load_model_and_scaler(model_path="models/best_model.joblib", scaler_path="models/scaler.joblib"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def preprocess_new(df, scaler):
    # basic: assume same columns as train (except target)
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    df[num_cols] = scaler.transform(df[num_cols])
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model.joblib")
    parser.add_argument("--scaler", default="models/scaler.joblib")
    parser.add_argument("--input", required=True, help="CSV file with same features as training data (no target)")
    parser.add_argument("--output", default="data/predictions_out.csv")
    args = parser.parse_args()

    model, scaler = load_model_and_scaler(args.model, args.scaler)
    df = pd.read_csv(args.input)
    # remove target if present
    if 'target' in df.columns:
        df = df.drop(columns=['target'])
    df_proc = preprocess_new(df, scaler)
    preds = model.predict(df_proc)
    probs = None
    try:
        probs = model.predict_proba(df_proc)[:,1]
    except:
        pass
    out = df.copy()
    out['y_pred'] = preds
    if probs is not None:
        out['y_proba'] = probs
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print("Saved predictions to", args.output)
