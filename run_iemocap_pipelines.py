import argparse
import ctypes
import gc
import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModel, AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor

from iemocap_pipeline_utils import (
    EmotionClassifier,
    build_iemocap_metadata,
    build_label_maps,
    compute_class_weights,
    create_dataloaders,
    create_train_val_test_dataloaders,
    embed_texts_with_distilbert,
    encode_labels,
    evaluate_model,
    extract_wave2vec_features_from_rows,
    filter_iemocap_metadata,
    load_iemocap_train_split,
    scale_feature_splits,
    scale_train_test,
    session_holdout_split,
    train_model,
    train_val_test_split,
    transcribe_rows_with_whisper,
)


DEFAULT_DATASET_ID = "AbstractTTS/IEMOCAP"
DEFAULT_LABEL_COLUMN = "major_emotion"
DEFAULT_DROP_LABELS = ("other",)
DEFAULT_TEST_SESSIONS = ("Ses05",)


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    use_audio: bool
    text_source: str
    wave2vec_model_name: str | None = None
    whisper_model_name: str | None = None
    text_model_name: str | None = None
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 1e-3
    dropout: float = 0.5
    validation_size: float | None = None
    early_stopping_patience: int | None = None
    monitor: str = "loss"


PIPELINE_CONFIGS = {
    "audio_only": PipelineConfig(
        name="audio_only",
        use_audio=True,
        text_source="none",
        wave2vec_model_name="facebook/wav2vec2-base-960h",
    ),
    "whisper_distilbert": PipelineConfig(
        name="whisper_distilbert",
        use_audio=False,
        text_source="whisper",
        whisper_model_name="base",
        text_model_name="distilbert-base-uncased",
    ),
    "transcript_distilbert": PipelineConfig(
        name="transcript_distilbert",
        use_audio=False,
        text_source="transcript",
        text_model_name="distilbert-base-uncased",
    ),
    "audio_transcript_fusion": PipelineConfig(
        name="audio_transcript_fusion",
        use_audio=True,
        text_source="transcript",
        wave2vec_model_name="facebook/wav2vec2-base-960h",
        text_model_name="distilbert-base-uncased",
        num_epochs=50,
        learning_rate=3e-4,
        dropout=0.3,
        validation_size=0.15,
        early_stopping_patience=8,
        monitor="loss",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run IEMOCAP training pipelines from the command line.")
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=["transcript_distilbert"],
        choices=list(PIPELINE_CONFIGS) + ["all"],
        help="One or more pipelines to run, or 'all'.",
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN)
    parser.add_argument("--drop-labels", nargs="*", default=list(DEFAULT_DROP_LABELS))
    parser.add_argument("--min-samples-per-class", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--test-sessions", nargs="+", default=list(DEFAULT_TEST_SESSIONS))
    parser.add_argument("--output-root", default="results/iemocap")
    parser.add_argument("--cache-dir", default="IEMOCAP_cache")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--validation-size", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--monitor", choices=["loss", "acc"], default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_pipelines(requested):
    if "all" in requested:
        return list(PIPELINE_CONFIGS)
    seen = []
    for name in requested:
        if name not in seen:
            seen.append(name)
    return seen


def build_run_root(output_root, run_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_name = run_name or timestamp
    run_root = Path(output_root) / root_name
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


def apply_overrides(config, args):
    updated = config

    if args.batch_size is not None:
        updated = replace(updated, batch_size=args.batch_size)
    if args.num_epochs is not None:
        updated = replace(updated, num_epochs=args.num_epochs)
    if args.learning_rate is not None:
        updated = replace(updated, learning_rate=args.learning_rate)
    if args.dropout is not None:
        updated = replace(updated, dropout=args.dropout)
    if args.validation_size is not None:
        updated = replace(updated, validation_size=args.validation_size)
    if args.early_stopping_patience is not None:
        updated = replace(updated, early_stopping_patience=args.early_stopping_patience)
    if args.monitor is not None:
        updated = replace(updated, monitor=args.monitor)

    return updated


def load_texts(metadata, text_source):
    if text_source == "none":
        return None
    if text_source == "transcript":
        return metadata["transcription"].fillna("").astype(str).str.strip().tolist()
    raise ValueError(f"Unsupported direct text source: {text_source}")


def clear_model_from_memory(*objects):
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_cache_tag(row_indices, *parts):
    digest = hashlib.md5()
    digest.update(np.asarray(row_indices, dtype=np.int64).tobytes())
    for part in parts:
        digest.update(str(part).encode("utf-8"))
    return digest.hexdigest()[:12]


def build_audio_features(dataset, row_indices, config, cache_dir, device):
    cache_tag = build_cache_tag(row_indices, config.wave2vec_model_name, "audio")
    cache_path = cache_dir / f"audio_wave2vec_features_{cache_tag}.npz"
    print(f"Loading Wave2Vec2 model: {config.wave2vec_model_name}")
    processor = Wav2Vec2Processor.from_pretrained(config.wave2vec_model_name)
    wave2vec_model = Wav2Vec2Model.from_pretrained(config.wave2vec_model_name).to(device)
    wave2vec_model.eval()

    features = extract_wave2vec_features_from_rows(
        dataset,
        row_indices,
        processor,
        wave2vec_model,
        device,
        cache_path=str(cache_path),
    )

    clear_model_from_memory(processor, wave2vec_model)
    return features


def build_whisper_texts(dataset, row_indices, config, cache_dir):
    import whisper

    cache_tag = build_cache_tag(row_indices, config.whisper_model_name, "whisper")
    cache_path = cache_dir / f"whisper_{config.whisper_model_name}_transcripts_{cache_tag}.npz"
    print(f"Loading Whisper model: {config.whisper_model_name}")
    whisper_model = whisper.load_model(config.whisper_model_name)
    texts = transcribe_rows_with_whisper(
        dataset,
        row_indices,
        whisper_model,
        cache_path=str(cache_path),
    )

    clear_model_from_memory(whisper_model)
    return texts


def build_text_features(texts, config, cache_dir, device):
    if texts is None:
        return None

    text_array = np.asarray(texts, dtype=object)
    text_digest = hashlib.md5()
    for text in text_array:
        text_digest.update(str(text).encode("utf-8"))
        text_digest.update(b"\0")
    cache_tag = text_digest.hexdigest()[:12]

    if config.text_source == "whisper":
        cache_path = cache_dir / f"whisper_distilbert_embeddings_{cache_tag}.npz"
    else:
        cache_path = cache_dir / f"provided_transcript_distilbert_embeddings_{cache_tag}.npz"

    print(f"Loading text model: {config.text_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    text_model = AutoModel.from_pretrained(config.text_model_name).to(device)
    text_model.eval()

    features = embed_texts_with_distilbert(
        texts,
        tokenizer,
        text_model,
        device,
        cache_path=str(cache_path),
        batch_size=16,
    )

    clear_model_from_memory(tokenizer, text_model)
    return features


def build_feature_matrix(dataset, metadata, config, cache_dir, device, output_dir):
    row_indices = metadata["row_idx"].to_numpy()
    feature_blocks = []
    texts = None

    if config.use_audio:
        audio_features = build_audio_features(dataset, row_indices, config, cache_dir, device)
        feature_blocks.append(audio_features)

    if config.text_source == "whisper":
        texts = build_whisper_texts(dataset, row_indices, config, cache_dir)
        whisper_frame = metadata[["file", "transcription"]].copy()
        whisper_frame["whisper_text"] = texts
        whisper_frame.to_csv(output_dir / "whisper_transcripts.csv", index=False)
    elif config.text_source == "transcript":
        texts = load_texts(metadata, config.text_source)

    if config.text_source != "none":
        text_features = build_text_features(texts, config, cache_dir, device)
        feature_blocks.append(text_features)

    if not feature_blocks:
        raise ValueError("Pipeline produced no feature blocks.")

    X = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    return X, texts


def plot_history(history, eval_name, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history[f"{eval_name}_loss"], label=eval_name)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history[f"{eval_name}_acc"], label=eval_name)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion(cm, label_names, title, output_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def history_to_frame(history):
    sequence_payload = {key: value for key, value in history.items() if isinstance(value, list)}
    return pd.DataFrame(sequence_payload)


def prepare_splits(X, y, metadata, config, args):
    label_names, label_to_idx, idx_to_label = build_label_maps(metadata[args.label_column])

    if config.validation_size is not None and config.validation_size > 0:
        train_mask, val_mask, test_mask = train_val_test_split(
            metadata,
            label_column=args.label_column,
            test_sessions=tuple(args.test_sessions),
            validation_size=config.validation_size,
            random_state=args.random_state,
        )

        X_train, X_val, X_test, scaler = scale_feature_splits(X[train_mask], X[val_mask], X[test_mask])
        y_train = y[train_mask]
        y_val = y[val_mask]
        y_test = y[test_mask]
        train_loader, val_loader, test_loader = create_train_val_test_dataloaders(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            batch_size=config.batch_size,
        )
        split_payload = {
            "train": metadata.loc[train_mask].reset_index().rename(columns={"index": "filtered_idx"}),
            "val": metadata.loc[val_mask].reset_index().rename(columns={"index": "filtered_idx"}),
            "test": metadata.loc[test_mask].reset_index().rename(columns={"index": "filtered_idx"}),
        }
        eval_loader = val_loader
        eval_name = "val"
    else:
        train_mask, test_mask = session_holdout_split(metadata, test_sessions=tuple(args.test_sessions))
        X_train, X_test, scaler = scale_train_test(X[train_mask], X[test_mask])
        y_train = y[train_mask]
        y_test = y[test_mask]
        train_loader, test_loader = create_dataloaders(
            X_train,
            y_train,
            X_test,
            y_test,
            batch_size=config.batch_size,
        )
        split_payload = {
            "train": metadata.loc[train_mask].reset_index().rename(columns={"index": "filtered_idx"}),
            "test": metadata.loc[test_mask].reset_index().rename(columns={"index": "filtered_idx"}),
        }
        eval_loader = test_loader
        eval_name = "test"

    class_weights, class_counts = compute_class_weights(y_train, num_classes=len(label_names))

    return {
        "label_names": label_names,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "scaler": scaler,
        "train_loader": train_loader,
        "eval_loader": eval_loader,
        "test_loader": test_loader,
        "class_weights": class_weights,
        "class_counts": class_counts,
        "split_payload": split_payload,
        "eval_name": eval_name,
        "y_test": y_test,
    }


def save_split_artifacts(output_dir, split_payload, label_column, label_names):
    split_counts = {}

    for split_name, frame in split_payload.items():
        frame.to_csv(output_dir / f"{split_name}_metadata.csv", index=False)
        split_counts[split_name] = frame[label_column].value_counts().reindex(label_names, fill_value=0)

    split_count_frame = pd.DataFrame(split_counts)
    split_count_frame.to_csv(output_dir / "split_counts.csv")


def save_predictions(output_dir, test_frame, all_labels, all_preds, label_names, texts=None):
    predictions = test_frame[["file", "session_id", "dialogue_id"]].copy()
    predictions["true_label"] = [label_names[idx] for idx in all_labels]
    predictions["pred_label"] = [label_names[idx] for idx in all_preds]
    predictions["correct"] = predictions["true_label"] == predictions["pred_label"]

    if "transcription" in test_frame.columns:
        predictions["transcription"] = test_frame["transcription"]
    if texts is not None:
        predictions["input_text"] = texts

    predictions.to_csv(output_dir / "predictions.csv", index=False)


def run_single_pipeline(pipeline_name, args, run_root, device):
    config = apply_overrides(PIPELINE_CONFIGS[pipeline_name], args)
    output_dir = run_root / pipeline_name
    output_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = Path(args.cache_dir) / "cli"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Running pipeline: {pipeline_name} ===")
    print(f"Results directory: {output_dir}")

    dataset = load_iemocap_train_split(args.dataset_id)
    metadata = build_iemocap_metadata(dataset)
    filtered_metadata, label_counts = filter_iemocap_metadata(
        metadata,
        label_column=args.label_column,
        drop_labels=set(args.drop_labels),
        min_samples_per_class=args.min_samples_per_class,
        max_samples=args.max_samples,
        random_state=args.random_state,
    )

    label_counts.rename("count").to_csv(output_dir / "label_counts.csv", header=True)

    X, texts = build_feature_matrix(dataset, filtered_metadata, config, cache_dir, device, output_dir)
    label_names, label_to_idx, idx_to_label = build_label_maps(filtered_metadata[args.label_column])
    y = encode_labels(filtered_metadata[args.label_column], label_to_idx)

    split_data = prepare_splits(X, y, filtered_metadata, config, args)
    save_split_artifacts(output_dir, split_data["split_payload"], args.label_column, split_data["label_names"])

    model = EmotionClassifier(
        input_size=X.shape[1],
        num_emotions=len(split_data["label_names"]),
        dropout_rate=config.dropout,
    ).to(device)

    history = train_model(
        model,
        split_data["train_loader"],
        split_data["eval_loader"],
        device,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        class_weights=split_data["class_weights"],
        eval_name=split_data["eval_name"],
        early_stopping_patience=config.early_stopping_patience,
        restore_best=config.validation_size is not None and config.validation_size > 0,
        monitor=config.monitor,
    )

    criterion = nn.CrossEntropyLoss(weight=split_data["class_weights"].to(device))
    test_loss, test_acc, all_preds, all_labels = evaluate_model(model, split_data["test_loader"], criterion, device)

    report_text = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(split_data["label_names"]))),
        target_names=split_data["label_names"],
        zero_division=0,
    )
    report_dict = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(split_data["label_names"]))),
        target_names=split_data["label_names"],
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(split_data["label_names"]))))

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as handle:
        handle.write(report_text)
        handle.write("\n")

    save_json(output_dir / "classification_report.json", report_dict)
    np.save(output_dir / "confusion_matrix.npy", cm)
    pd.DataFrame(cm, index=split_data["label_names"], columns=split_data["label_names"]).to_csv(
        output_dir / "confusion_matrix.csv"
    )

    plot_history(history, split_data["eval_name"], output_dir / "history.png")
    plot_confusion(
        cm,
        split_data["label_names"],
        f"Confusion Matrix - {pipeline_name}",
        output_dir / "confusion_matrix.png",
    )

    history_frame = history_to_frame(history)
    history_frame.to_csv(output_dir / "history.csv", index=False)
    save_json(output_dir / "history.json", history)

    test_frame = split_data["split_payload"]["test"]
    test_texts = None
    if texts is not None:
        test_texts = np.asarray(texts, dtype=object)[test_frame["filtered_idx"].to_numpy()]
    save_predictions(output_dir, test_frame, all_labels, all_preds, split_data["label_names"], texts=test_texts)

    metrics = {
        "pipeline": pipeline_name,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "num_labels": len(split_data["label_names"]),
        "num_samples_total": int(len(filtered_metadata)),
        "num_samples_test": int(len(test_frame)),
        "feature_dim": int(X.shape[1]),
        "device": str(device),
        "best_epoch": history.get("best_epoch"),
        f"best_{split_data['eval_name']}_{config.monitor}": history.get(
            f"best_{split_data['eval_name']}_{config.monitor}"
        ),
    }
    save_json(output_dir / "metrics.json", metrics)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "label_names": split_data["label_names"],
        "label_to_idx": split_data["label_to_idx"],
        "idx_to_label": split_data["idx_to_label"],
        "scaler_mean": split_data["scaler"].mean_,
        "scaler_scale": split_data["scaler"].scale_,
        "config": {
            "dataset_id": args.dataset_id,
            "label_column": args.label_column,
            "drop_labels": list(args.drop_labels),
            "min_samples_per_class": args.min_samples_per_class,
            "max_samples": args.max_samples,
            "test_sessions": list(args.test_sessions),
            "random_state": args.random_state,
            **asdict(config),
        },
        "history": history,
        "metrics": metrics,
    }
    torch.save(checkpoint, output_dir / "checkpoint.pth")

    save_json(
        output_dir / "run_config.json",
        {
            "dataset_id": args.dataset_id,
            "label_column": args.label_column,
            "drop_labels": list(args.drop_labels),
            "min_samples_per_class": args.min_samples_per_class,
            "max_samples": args.max_samples,
            "test_sessions": list(args.test_sessions),
            "random_state": args.random_state,
            "device": str(device),
            **asdict(config),
        },
    )

    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Saved artifacts to: {output_dir}")

    clear_model_from_memory(model)
    return {
        "pipeline": pipeline_name,
        "output_dir": str(output_dir),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "feature_dim": int(X.shape[1]),
    }


def main():
    args = parse_args()
    pipelines = resolve_pipelines(args.pipelines)
    run_root = build_run_root(args.output_root, args.run_name)
    device = resolve_device(args.device)

    save_json(
        run_root / "run_manifest.json",
        {
            "created_at": datetime.now().isoformat(),
            "pipelines": pipelines,
            "dataset_id": args.dataset_id,
            "label_column": args.label_column,
            "drop_labels": list(args.drop_labels),
            "min_samples_per_class": args.min_samples_per_class,
            "max_samples": args.max_samples,
            "test_sessions": list(args.test_sessions),
            "cache_dir": args.cache_dir,
            "device": str(device),
        },
    )

    summary_rows = []
    for pipeline_name in pipelines:
        summary_rows.append(run_single_pipeline(pipeline_name, args, run_root, device))

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(run_root / "summary.csv", index=False)
    save_json(run_root / "summary.json", summary_rows)

    print("\n=== All requested pipelines completed ===")
    print(summary_frame.to_string(index=False))
    print(f"\nRun root: {run_root}")


if __name__ == "__main__":
    main()
