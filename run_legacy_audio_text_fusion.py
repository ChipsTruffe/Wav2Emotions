from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from config import MELD_FOLDER, MODEL_CONFIG, RAVDESS_FOLDER
from legacy_fusion_utils import (
    DEFAULT_TEXT_MODEL,
    DEFAULT_WHISPER_MODEL,
    SAFE_WAVE2VEC2_MODEL,
    build_cache_dir,
    build_cache_tag,
    build_run_dir,
    clear_objects,
    compute_class_weights,
    embed_texts_with_distilbert,
    encode_label_series,
    extract_audio_feature_matrix,
    load_distilbert_stack,
    load_meld_metadata,
    load_ravdess_metadata,
    load_wave2vec_stack,
    preload_conda_runtime,
    resolve_device,
    save_json,
    save_training_artifacts,
    set_random_seeds,
    split_and_scale_features,
    transcribe_audio_paths,
)
from models import EMOTION_LABELS, EmotionClassifier, create_dataloaders, evaluate, train_model


preload_conda_runtime()


@dataclass(frozen=True)
class FusionConfig:
    dataset: str
    data_folder: str
    meld_split: str
    output_root: str
    cache_root: str
    run_name: str | None
    wave2vec_model_name: str
    text_model_name: str
    whisper_model_name: str
    batch_size: int
    num_epochs: int
    learning_rate: float
    dropout: float
    test_size: float
    random_state: int
    max_samples: int | None
    device: str
    use_class_weights: bool


def parse_args(default_dataset=None):
    parser = argparse.ArgumentParser(description="Run audio+text fusion on RAVDESS or MELD.")
    parser.add_argument("--dataset", choices=["ravdess", "meld"], required=default_dataset is None, default=default_dataset)
    parser.add_argument("--data-folder", default=None)
    parser.add_argument("--meld-split", default="train")
    parser.add_argument("--output-root", default="results/legacy_fusion")
    parser.add_argument("--cache-root", default="results/legacy_fusion_cache")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wave2vec-model-name", default=SAFE_WAVE2VEC2_MODEL)
    parser.add_argument("--text-model-name", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--whisper-model-name", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--batch-size", type=int, default=MODEL_CONFIG["batch_size"])
    parser.add_argument("--num-epochs", type=int, default=MODEL_CONFIG["num_epochs"])
    parser.add_argument("--learning-rate", type=float, default=MODEL_CONFIG["learning_rate"])
    parser.add_argument("--dropout", type=float, default=MODEL_CONFIG["dropout"])
    parser.add_argument("--test-size", type=float, default=MODEL_CONFIG["test_size"])
    parser.add_argument("--random-state", type=int, default=MODEL_CONFIG["random_state"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Apply inverse-frequency class weights during training.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weights even if the dataset default would enable them.",
    )
    return parser.parse_args()


def build_config(args):
    default_folder = RAVDESS_FOLDER if args.dataset == "ravdess" else MELD_FOLDER
    use_class_weights = args.dataset == "meld"
    if args.use_class_weights:
        use_class_weights = True
    if args.no_class_weights:
        use_class_weights = False

    return FusionConfig(
        dataset=args.dataset,
        data_folder=args.data_folder or default_folder,
        meld_split=args.meld_split,
        output_root=args.output_root,
        cache_root=args.cache_root,
        run_name=args.run_name,
        wave2vec_model_name=args.wave2vec_model_name,
        text_model_name=args.text_model_name,
        whisper_model_name=args.whisper_model_name,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        test_size=args.test_size,
        random_state=args.random_state,
        max_samples=args.max_samples,
        device=args.device,
        use_class_weights=use_class_weights,
    )


def load_metadata(config):
    if config.dataset == "ravdess":
        return load_ravdess_metadata(config.data_folder)
    return load_meld_metadata(config.data_folder, split=config.meld_split, max_samples=config.max_samples)


def build_audio_features(metadata, config, cache_dir, device):
    cache_tag = build_cache_tag(metadata["audio_path"].tolist(), config.wave2vec_model_name, "audio")
    cache_path = cache_dir / f"audio_features_{cache_tag}.npz"

    processor, wave2vec_model = load_wave2vec_stack(config.wave2vec_model_name, device)
    X_audio = extract_audio_feature_matrix(metadata, processor, wave2vec_model, device, cache_path=str(cache_path))
    clear_objects(processor, wave2vec_model)
    return X_audio


def build_texts(metadata, config, cache_dir, device):
    if config.dataset == "meld":
        return metadata["text"].fillna("").astype(str).str.strip().tolist()

    import whisper

    cache_tag = build_cache_tag(metadata["audio_path"].tolist(), config.whisper_model_name, "whisper")
    cache_path = cache_dir / f"whisper_texts_{cache_tag}.npz"
    whisper_model = whisper.load_model(config.whisper_model_name, device=device.type)
    texts = transcribe_audio_paths(metadata["audio_path"].tolist(), whisper_model, cache_path=str(cache_path))
    clear_objects(whisper_model)
    return texts


def build_text_features(texts, config, cache_dir, device):
    cache_tag = build_cache_tag(texts, config.text_model_name, "text")
    cache_path = cache_dir / f"text_features_{cache_tag}.npz"

    tokenizer, text_model = load_distilbert_stack(config.text_model_name, device)
    X_text = embed_texts_with_distilbert(texts, tokenizer, text_model, device, cache_path=str(cache_path), batch_size=16)
    clear_objects(tokenizer, text_model)
    return X_text


def main(default_dataset=None):
    config = build_config(parse_args(default_dataset=default_dataset))
    device = resolve_device(config.device)
    set_random_seeds(config.random_state)

    output_dir = build_run_dir(config.output_root, config.dataset, config.run_name)
    cache_dir = build_cache_dir(config.cache_root, config.dataset)

    print(f"Dataset: {config.dataset}")
    print(f"Data folder: {config.data_folder}")
    print(f"Output directory: {output_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Device: {device}")

    metadata = load_metadata(config)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    print(f"Loaded {len(metadata)} samples")

    print("Extracting audio features...")
    X_audio = build_audio_features(metadata, config, cache_dir, device)

    print("Preparing text inputs...")
    texts = build_texts(metadata, config, cache_dir, device)
    metadata = metadata.copy()
    metadata["text"] = texts
    metadata.to_csv(output_dir / "metadata_with_text.csv", index=False)

    print("Embedding texts...")
    X_text = build_text_features(texts, config, cache_dir, device)

    X = np.concatenate([X_audio.astype(np.float32), X_text.astype(np.float32)], axis=1)
    y = encode_label_series(metadata["emotion"].tolist())

    X_train, X_test, y_train, y_test, train_idx, test_idx, scaler = split_and_scale_features(
        X,
        y,
        random_state=config.random_state,
        test_size=config.test_size,
    )
    train_loader, test_loader = create_dataloaders(X_train, X_test, y_train, y_test, batch_size=config.batch_size)

    class_weights = None
    if config.use_class_weights:
        class_weights, class_counts = compute_class_weights(y_train)
        class_count_payload = {EMOTION_LABELS[idx]: int(count) for idx, count in enumerate(class_counts) if count > 0}
        save_json(output_dir / "class_counts.json", class_count_payload)

    model = EmotionClassifier(
        input_size=X_train.shape[1],
        num_emotions=len(EMOTION_LABELS),
        dropout_rate=config.dropout,
    ).to(device)

    train_losses, train_accs, test_losses, test_accs = train_model(
        model,
        train_loader,
        test_loader,
        device,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        class_weights=class_weights,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) if class_weights is not None else nn.CrossEntropyLoss()
    final_loss, final_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)

    run_config = asdict(config)
    run_config["resolved_device"] = str(device)
    run_config["feature_dim"] = int(X.shape[1])
    run_config["num_samples"] = int(len(metadata))

    metrics = save_training_artifacts(
        output_dir=output_dir,
        metadata=metadata,
        train_idx=train_idx,
        test_idx=test_idx,
        y_train=y_train,
        y_test=y_test,
        train_losses=train_losses,
        train_accs=train_accs,
        test_losses=test_losses,
        test_accs=test_accs,
        all_preds=all_preds,
        all_labels=all_labels,
        final_acc=final_acc,
        final_loss=final_loss,
        scaler=scaler,
        model=model,
        run_config=run_config,
        title_prefix=f"{config.dataset.upper()} Audio+Text Fusion",
    )

    print(f"Final test accuracy: {metrics['test_accuracy']:.2f}%")
    print(f"Final test loss: {metrics['test_loss']:.4f}")
    print(f"Saved artifacts to: {output_dir}")

    clear_objects(model)


if __name__ == "__main__":
    main()
