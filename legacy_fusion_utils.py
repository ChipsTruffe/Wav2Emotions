from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

def preload_conda_runtime():
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return

    for library_name in ("libgcc_s.so.1", "libstdc++.so.6"):
        library_path = os.path.join(prefix, "lib", library_name)
        if os.path.exists(library_path):
            ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)


preload_conda_runtime()

import librosa
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from config import EMOTION_DICT, MELD_EMOTION_LIST, MELD_EMOTIONS, MELD_FOLDER, MODEL_CONFIG, RAVDESS_FOLDER
from models import EMOTION_LABELS, EMOTION_TO_IDX, EmotionClassifier, create_dataloaders, evaluate, train_model
from utils import extract_wave2vec_features


matplotlib.use("Agg")
import matplotlib.pyplot as plt


SAFE_WAVE2VEC2_MODEL = "facebook/wav2vec2-base-960h"
DEFAULT_TEXT_MODEL = "distilbert-base-uncased"
DEFAULT_WHISPER_MODEL = "base"


class ArrayScaler:
    def __init__(self, mean_, scale_):
        self.mean_ = mean_
        self.scale_ = scale_

    def transform(self, X):
        return ((X - self.mean_) / self.scale_).astype(np.float32)


def set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def stratified_train_test_indices(y, test_size, random_state):
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    train_indices = []
    test_indices = []

    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label)
        shuffled = rng.permutation(label_indices)

        if len(label_indices) == 1:
            n_test = 1
        else:
            n_test = int(round(len(label_indices) * test_size))
            n_test = max(1, min(len(label_indices) - 1, n_test))

        test_indices.extend(shuffled[:n_test])
        train_indices.extend(shuffled[n_test:])

    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def fit_array_scaler(X_train):
    mean_ = np.mean(X_train, axis=0, dtype=np.float64)
    scale_ = np.std(X_train, axis=0, dtype=np.float64)
    scale_[scale_ == 0] = 1.0
    return ArrayScaler(mean_.astype(np.float32), scale_.astype(np.float32))


def confusion_matrix_array(all_labels, all_preds, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(all_labels, all_preds):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def classification_report_dict(all_labels, all_preds, target_names):
    cm = confusion_matrix_array(all_labels, all_preds, len(target_names))
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    total = int(cm.sum())

    report = {}
    precisions = []
    recalls = []
    f1s = []
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for idx, name in enumerate(target_names):
        tp = float(cm[idx, idx])
        fp = float(predicted[idx] - cm[idx, idx])
        fn = float(support[idx] - cm[idx, idx])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        supp = int(support[idx])

        report[name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": supp,
        }

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        weighted_precision += precision * supp
        weighted_recall += recall * supp
        weighted_f1 += f1 * supp

    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0
    macro_count = len(target_names) if target_names else 1
    report["accuracy"] = accuracy
    report["macro avg"] = {
        "precision": float(sum(precisions) / macro_count),
        "recall": float(sum(recalls) / macro_count),
        "f1-score": float(sum(f1s) / macro_count),
        "support": total,
    }
    report["weighted avg"] = {
        "precision": float(weighted_precision / total) if total > 0 else 0.0,
        "recall": float(weighted_recall / total) if total > 0 else 0.0,
        "f1-score": float(weighted_f1 / total) if total > 0 else 0.0,
        "support": total,
    }
    return report


def classification_report_text(report_dict, target_names):
    header = f"{'':14s}{'precision':>10s}{'recall':>10s}{'f1-score':>10s}{'support':>10s}"
    lines = [header, ""]

    for name in target_names:
        row = report_dict[name]
        lines.append(
            f"{name:14s}{row['precision']:10.2f}{row['recall']:10.2f}{row['f1-score']:10.2f}{row['support']:10d}"
        )

    lines.append("")
    accuracy = report_dict["accuracy"]
    total_support = report_dict["weighted avg"]["support"]
    lines.append(f"{'accuracy':14s}{'':20s}{accuracy:10.2f}{total_support:10d}")

    for summary_name in ("macro avg", "weighted avg"):
        row = report_dict[summary_name]
        lines.append(
            f"{summary_name:14s}{row['precision']:10.2f}{row['recall']:10.2f}{row['f1-score']:10.2f}{row['support']:10d}"
        )

    return "\n".join(lines)


def build_run_dir(output_root, dataset_name, run_name=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(output_root) / dataset_name / (run_name or timestamp)
    root.mkdir(parents=True, exist_ok=False)
    return root


def build_cache_dir(cache_root, dataset_name):
    cache_dir = Path(cache_root) / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def clear_objects(*objects):
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_cache_tag(items, *parts):
    digest = hashlib.md5()
    for item in items:
        digest.update(str(item).encode("utf-8"))
        digest.update(b"\0")
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def aggregate_token_embeddings(hidden_states, attention_mask):
    valid_tokens = hidden_states[attention_mask.bool()]
    mean_emb = torch.mean(valid_tokens, dim=0)
    std_emb = torch.std(valid_tokens, dim=0)
    max_emb = torch.max(valid_tokens, dim=0)[0]
    min_emb = torch.min(valid_tokens, dim=0)[0]
    return torch.cat([mean_emb, std_emb, max_emb, min_emb]).cpu().numpy().astype(np.float32)


def load_ravdess_metadata(data_folder=RAVDESS_FOLDER):
    records = []
    data_folder = Path(data_folder)

    if not data_folder.exists():
        raise FileNotFoundError(f"RAVDESS folder not found: {data_folder}")

    for audio_path in sorted(data_folder.rglob("*.wav")):
        parts = audio_path.name.split("-")
        if len(parts) < 3 or parts[2] not in EMOTION_DICT:
            continue

        records.append(
            {
                "audio_path": str(audio_path),
                "emotion": EMOTION_DICT[parts[2]],
                "file_name": audio_path.name,
                "actor_id": parts[-1].replace(".wav", ""),
                "text": "",
            }
        )

    metadata = pd.DataFrame(records)
    if metadata.empty:
        raise RuntimeError(f"No valid RAVDESS audio files found under {data_folder}")
    return metadata


def load_meld_metadata(data_folder=MELD_FOLDER, split="train", max_samples=None):
    data_folder = Path(data_folder)
    csv_path = data_folder / f"{split}.csv"
    audio_dir = data_folder / split

    if not csv_path.exists():
        raise FileNotFoundError(f"MELD CSV not found: {csv_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"MELD audio folder not found: {audio_dir}")

    frame = pd.read_csv(csv_path, encoding="latin-1")
    records = []

    for _, row in frame.iterrows():
        emotion = str(row["Emotion"]).strip().lower()
        if emotion not in MELD_EMOTION_LIST:
            continue

        dia_id = int(row["Dialogue_ID"])
        utt_id = int(row["Utterance_ID"])
        audio_path = audio_dir / f"dia{dia_id}_utt{utt_id}.flac"
        if not audio_path.exists():
            continue

        records.append(
            {
                "audio_path": str(audio_path),
                "emotion": MELD_EMOTIONS[emotion],
                "dialogue_id": dia_id,
                "utterance_id": utt_id,
                "speaker": str(row["Speaker"]) if "Speaker" in row else "",
                "file_name": audio_path.name,
                "text": str(row["Utterance"]).strip() if "Utterance" in row and pd.notna(row["Utterance"]) else "",
            }
        )

        if max_samples is not None and len(records) >= max_samples:
            break

    metadata = pd.DataFrame(records)
    if metadata.empty:
        raise RuntimeError(f"No valid MELD samples found under {data_folder} for split={split}")
    return metadata


def extract_audio_feature_matrix(metadata, processor, wave2vec_model, device, cache_path=None):
    audio_paths = metadata["audio_path"].tolist()

    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        if np.array_equal(cached["audio_paths"], np.asarray(audio_paths, dtype=object)):
            print(f"Loaded cached audio features from {cache_path}")
            return cached["X"].astype(np.float32)

    features = []
    for audio_path in audio_paths:
        features.append(extract_wave2vec_features(audio_path, processor, wave2vec_model, device))

    X = np.asarray(features, dtype=np.float32)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            audio_paths=np.asarray(audio_paths, dtype=object),
            X=X,
        )
        print(f"Saved audio features to {cache_path}")

    return X


def transcribe_audio_paths(audio_paths, whisper_model, cache_path=None):
    audio_paths = [str(path) for path in audio_paths]

    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        if np.array_equal(cached["audio_paths"], np.asarray(audio_paths, dtype=object)):
            print(f"Loaded cached transcriptions from {cache_path}")
            return cached["texts"].tolist()

    transcripts = []
    use_fp16 = getattr(getattr(whisper_model, "device", torch.device("cpu")), "type", "cpu") == "cuda"

    for audio_path in audio_paths:
        waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
        result = whisper_model.transcribe(
            waveform,
            language="en",
            task="transcribe",
            verbose=False,
            fp16=use_fp16,
        )
        transcripts.append(result["text"].strip())

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            audio_paths=np.asarray(audio_paths, dtype=object),
            texts=np.asarray(transcripts, dtype=object),
        )
        print(f"Saved transcriptions to {cache_path}")

    return transcripts


def embed_texts_with_distilbert(texts, tokenizer, text_model, device, cache_path=None, batch_size=16, max_length=128):
    text_array = np.asarray(texts, dtype=object)

    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        if np.array_equal(cached["texts"], text_array):
            print(f"Loaded cached text embeddings from {cache_path}")
            return cached["X"].astype(np.float32)

    text_model.eval()
    embeddings = []

    for start in range(0, len(texts), batch_size):
        batch_texts = [text.strip() if str(text).strip() else "[EMPTY]" for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            hidden_states = text_model(**encoded).last_hidden_state

        attention_mask = encoded["attention_mask"]
        for row_idx in range(hidden_states.shape[0]):
            embeddings.append(aggregate_token_embeddings(hidden_states[row_idx], attention_mask[row_idx]))

    X = np.asarray(embeddings, dtype=np.float32)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, texts=text_array, X=X)
        print(f"Saved text embeddings to {cache_path}")

    return X


def load_wave2vec_stack(model_name, device):
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
    from transformers.models.wav2vec2.processing_wav2vec2 import Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(model_name, token=False)
    model = Wav2Vec2Model.from_pretrained(model_name, token=False).to(device)
    model.eval()
    return processor, model


def load_distilbert_stack(model_name, device):
    from transformers.models.distilbert.modeling_distilbert import DistilBertModel
    try:
        from transformers.models.distilbert.tokenization_distilbert_fast import DistilBertTokenizerFast as TokenizerCls
    except ModuleNotFoundError:
        from transformers.models.distilbert.tokenization_distilbert import DistilBertTokenizer as TokenizerCls

    tokenizer = TokenizerCls.from_pretrained(model_name, token=False)
    model = DistilBertModel.from_pretrained(model_name, token=False).to(device)
    model.eval()
    return tokenizer, model


def encode_label_series(labels):
    return np.asarray([EMOTION_TO_IDX[label] for label in labels], dtype=np.int64)


def compute_class_weights(y_train):
    counts = np.bincount(y_train, minlength=len(EMOTION_LABELS))
    weights = np.zeros(len(EMOTION_LABELS), dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = len(y_train) / (len(counts) * counts[nonzero])
    return torch.tensor(weights, dtype=torch.float32), counts


def split_and_scale_features(X, y, random_state, test_size):
    train_idx, test_idx = stratified_train_test_indices(y, test_size=test_size, random_state=random_state)

    scaler = fit_array_scaler(X[train_idx])
    X_train = scaler.transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    return X_train, X_test, y[train_idx], y[test_idx], train_idx, test_idx, scaler


def plot_history(train_losses, train_accs, test_losses, test_accs, output_path, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label="Train Loss", linewidth=2)
    axes[0].plot(test_losses, label="Test Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title_prefix} Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_accs, label="Train Accuracy", linewidth=2)
    axes[1].plot(test_accs, label="Test Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"{title_prefix} Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(all_labels, all_preds, output_path, title):
    cm = confusion_matrix_array(all_labels, all_preds, len(EMOTION_LABELS))
    cm_frame = pd.DataFrame(cm, index=EMOTION_LABELS, columns=EMOTION_LABELS)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_frame, annot=True, fmt="d", cmap="Blues", xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.title(title)
    plt.ylabel("True Emotion")
    plt.xlabel("Predicted Emotion")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return cm_frame


def save_training_artifacts(
    output_dir,
    metadata,
    train_idx,
    test_idx,
    y_train,
    y_test,
    train_losses,
    train_accs,
    test_losses,
    test_accs,
    all_preds,
    all_labels,
    final_acc,
    final_loss,
    scaler,
    model,
    run_config,
    title_prefix,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_metadata = metadata.iloc[train_idx].reset_index(drop=True)
    test_metadata = metadata.iloc[test_idx].reset_index(drop=True)
    train_metadata.to_csv(output_dir / "train_metadata.csv", index=False)
    test_metadata.to_csv(output_dir / "test_metadata.csv", index=False)

    history = {
        "train_loss": [float(x) for x in train_losses],
        "train_acc": [float(x) for x in train_accs],
        "test_loss": [float(x) for x in test_losses],
        "test_acc": [float(x) for x in test_accs],
    }
    save_json(output_dir / "history.json", history)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    plot_history(train_losses, train_accs, test_losses, test_accs, output_dir / "history.png", title_prefix)

    report_json = classification_report_dict(all_labels, all_preds, EMOTION_LABELS)
    criterion_report = classification_report_text(report_json, EMOTION_LABELS)
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as handle:
        handle.write(criterion_report)
        handle.write("\n")

    save_json(output_dir / "classification_report.json", report_json)

    cm_frame = plot_confusion_matrix(all_labels, all_preds, output_dir / "confusion_matrix.png", title_prefix)
    cm_frame.to_csv(output_dir / "confusion_matrix.csv")
    np.save(output_dir / "confusion_matrix.npy", cm_frame.to_numpy())

    predictions = test_metadata.copy()
    predictions["true_emotion"] = [EMOTION_LABELS[idx] for idx in all_labels]
    predictions["pred_emotion"] = [EMOTION_LABELS[idx] for idx in all_preds]
    predictions["correct"] = predictions["true_emotion"] == predictions["pred_emotion"]
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    metrics = {
        "test_accuracy": float(final_acc),
        "test_loss": float(final_loss),
        "num_samples_total": int(len(metadata)),
        "num_samples_train": int(len(train_idx)),
        "num_samples_test": int(len(test_idx)),
        "feature_dim": int(model.fc1.in_features),
    }
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "run_config.json", run_config)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "history": history,
        "metrics": metrics,
        "config": run_config,
        "emotion_labels": EMOTION_LABELS,
    }
    torch.save(checkpoint, output_dir / "checkpoint.pth")

    return metrics
