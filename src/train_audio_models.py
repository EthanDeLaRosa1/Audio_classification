"""
Audio Classification: Bird vs Cat vs Dog

Usage:
    python src/train_audio_models.py --data-dir Animals --out-dir results

Expected data layout:
    Animals/
      bird/*.wav
      cat/*.wav
      dog/*.wav
"""
import argparse, glob, json, os, warnings
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def read_wav(path):
    sr, y = wavfile.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    maxv = np.max(np.abs(y)) if len(y) else 1
    if maxv > 0:
        y = y / maxv
    return sr, y


def frame_signal(y, frame=512, hop=256):
    if len(y) < frame:
        y = np.pad(y, (0, frame - len(y)))
    n_frames = 1 + (len(y) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    return y[idx]


def extract_features(path):
    sr, y = read_wav(path)
    if len(y) == 0:
        return {}
    duration = len(y) / sr
    frames = frame_signal(y)
    win = get_window("hann", frames.shape[1], fftbins=True)
    wf = frames * win
    mag = np.abs(rfft(wf, axis=1)) + 1e-12
    freqs = rfftfreq(wf.shape[1], 1 / sr)
    power = mag ** 2
    psum = power.sum(axis=1) + 1e-12

    centroid = (power * freqs).sum(axis=1) / psum
    bandwidth = np.sqrt(((freqs - centroid[:, None]) ** 2 * power).sum(axis=1) / psum)
    csum = np.cumsum(power, axis=1)
    rolloff = freqs[np.argmax(csum >= (0.85 * psum)[:, None], axis=1)]
    pnorm = power / psum[:, None]
    entropy = -(pnorm * np.log2(pnorm + 1e-12)).sum(axis=1)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    avg_power = power.mean(axis=0)
    total = avg_power.sum() + 1e-12
    logspec = np.log(avg_power + 1e-12)
    cep = np.real(np.fft.rfft(logspec, n=64))[:10]

    feats = {
        "duration": duration,
        "rms_mean": rms.mean(), "rms_std": rms.std(),
        "zcr_mean": zcr.mean(), "zcr_std": zcr.std(),
        "centroid_mean": centroid.mean(), "centroid_std": centroid.std(),
        "bandwidth_mean": bandwidth.mean(), "bandwidth_std": bandwidth.std(),
        "rolloff85_mean": rolloff.mean(), "rolloff85_std": rolloff.std(),
        "entropy_mean": entropy.mean(), "entropy_std": entropy.std(),
        "dom_freq": freqs[np.argmax(avg_power)],
        "low_band": avg_power[(freqs >= 0) & (freqs < 1000)].sum() / total,
        "mid_band": avg_power[(freqs >= 1000) & (freqs < 3000)].sum() / total,
        "high_band": avg_power[freqs >= 3000].sum() / total,
    }
    for i, c in enumerate(cep):
        feats[f"cep{i+1}"] = c
    return {k: float(v) for k, v in feats.items()}


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for label in sorted(os.listdir(data_dir)):
        folder = os.path.join(data_dir, label)
        if not os.path.isdir(folder):
            continue
        for wav_path in glob.glob(os.path.join(folder, "*.wav")):
            feat = extract_features(wav_path)
            feat.update({"label": label, "file": os.path.basename(wav_path)})
            rows.append(feat)

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ["label", "file"]]
    missing_before = int(df[feature_cols].isna().sum().sum())

    # Outlier handling: cap numeric features at the 1st/99th percentiles.
    for c in feature_cols:
        lo, hi = df[c].quantile([0.01, 0.99])
        df[c] = df[c].clip(lo, hi)

    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    X = df[feature_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "KNN": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=7))]),
        "Logistic Regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1500, class_weight="balanced"))]),
        "Random Forest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"))]),
        "XGBoost": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", XGBClassifier(n_estimators=180, learning_rate=0.05, max_depth=3, subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss", random_state=42, n_jobs=2))]),
        "Neural Network": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(48, 24), early_stopping=True, max_iter=400, random_state=42))]),
    }

    results, preds = [], {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        preds[name] = pred
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Macro F1": f1_score(y_test, pred, average="macro"),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "R²": r2_score(y_test, pred),
        })

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    results_df.to_csv(os.path.join(out_dir, "model_results.csv"), index=False)
    df.to_csv(os.path.join(out_dir, "audio_features.csv"), index=False)

    best = results_df.iloc[0]["Model"]
    cm = confusion_matrix(y_test, preds[best])
    summary = {
        "n_samples": int(len(df)),
        "class_counts": df["label"].value_counts().sort_index().to_dict(),
        "n_features": len(feature_cols),
        "labels": list(le.classes_),
        "best_model": best,
        "best_accuracy": float(results_df.iloc[0]["Accuracy"]),
        "best_macro_f1": float(results_df.iloc[0]["Macro F1"]),
        "missing_values_found": missing_before,
        "outlier_method": "numeric features clipped to 1st/99th percentiles",
        "categorical_method": "target label encoded using sklearn LabelEncoder",
        "imbalance_note": "classes are close in size; used a stratified split and balanced class weights where supported",
        "classification_report": classification_report(y_test, preds[best], target_names=le.classes_, output_dict=True),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    counts = df["label"].value_counts().sort_index()
    plt.figure(figsize=(7, 4.5)); plt.bar(counts.index, counts.values); plt.title("Dataset class balance"); plt.ylabel("Audio clips"); plt.xlabel("Class"); savefig(os.path.join(out_dir, "class_balance.png"))
    order = results_df.sort_values("Accuracy")
    plt.figure(figsize=(9, 4.8)); plt.barh(order["Model"], order["Accuracy"]); plt.title("Model comparison: accuracy"); plt.xlabel("Accuracy on holdout test set"); plt.xlim(0, 1); savefig(os.path.join(out_dir, "model_accuracy.png"))
    order = results_df.sort_values("RMSE", ascending=False)
    plt.figure(figsize=(9, 4.8)); plt.barh(order["Model"], order["RMSE"]); plt.title("Model comparison: RMSE on encoded labels"); plt.xlabel("RMSE (lower is better)"); savefig(os.path.join(out_dir, "model_rmse.png"))
    order = results_df.sort_values("R²")
    plt.figure(figsize=(9, 4.8)); plt.barh(order["Model"], order["R²"]); plt.title("Model comparison: R² on encoded labels"); plt.xlabel("R² (higher is better)"); savefig(os.path.join(out_dir, "model_r2.png"))
    plt.figure(figsize=(5.5, 4.8)); plt.imshow(cm); plt.title(f"Confusion matrix: {best}"); plt.xticks(np.arange(len(le.classes_)), le.classes_); plt.yticks(np.arange(len(le.classes_)), le.classes_); plt.xlabel("Predicted"); plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.colorbar(fraction=0.046, pad=0.04); savefig(os.path.join(out_dir, "confusion_matrix.png"))
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Animals")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)
