"""Predict the crop for a soil/climate row with the saved models.

Only SVM was trained on MinMax-scaled features; the other four were trained on
raw values. Mixing that up drops every model to chance, so the mapping lives
here in one place.

    python predict.py --model "Random Forest" 90 42 43 20.9 82 6.5 202.9
    python predict.py --all          # score every model on modified_dataset.csv
"""
import argparse
import glob
import os
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
NEEDS_SCALING = {"SVM"}
HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    return joblib.load(os.path.join(HERE, f"{name}.pkl"))


def predict(model_name, values):
    x = np.asarray(values, dtype=float).reshape(1, -1)
    if model_name in NEEDS_SCALING:
        x = load("scaler").transform(x)
    model = load(model_name)
    return load("label_encoder").inverse_transform(model.predict(x))[0]


def score_all():
    df = pd.read_csv(os.path.join(HERE, "modified_dataset.csv"))
    X, y = df[FEATURES].to_numpy(dtype=float), df["label"].to_numpy()
    encoder = load("label_encoder")
    scaler = load("scaler")
    for path in sorted(glob.glob(os.path.join(HERE, "*.pkl"))):
        name = os.path.basename(path)[:-4]
        if name in ("scaler", "label_encoder"):
            continue
        data = scaler.transform(X) if name in NEEDS_SCALING else X
        acc = (encoder.inverse_transform(joblib.load(path).predict(data)) == y).mean()
        print(f"{name:22} {acc * 100:5.1f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Random Forest")
    p.add_argument("--all", action="store_true", help="score every model on the dataset")
    p.add_argument("values", nargs="*", type=float, help=f"{len(FEATURES)} values: {', '.join(FEATURES)}")
    args = p.parse_args()

    if args.all:
        score_all()
    elif len(args.values) == len(FEATURES):
        print(predict(args.model, args.values))
    else:
        p.error(f"give {len(FEATURES)} values in this order: {', '.join(FEATURES)}")


if __name__ == "__main__":
    main()
