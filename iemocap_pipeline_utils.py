import io
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Audio, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DEFAULT_DATASET_ID = "AbstractTTS/IEMOCAP"
DEFAULT_SAMPLE_RATE = 16000


def load_iemocap_train_split(dataset_id=DEFAULT_DATASET_ID):
    dataset_dict = load_dataset(dataset_id)
    dataset_dict = dataset_dict.cast_column("audio", Audio(decode=False))
    return dataset_dict["train"]


def build_iemocap_metadata(dataset):
    metadata = dataset.remove_columns(["audio"]).to_pandas().copy()
    metadata["row_idx"] = np.arange(len(metadata))
    metadata["session_id"] = metadata["file"].str.extract(r"(Ses\d{2})", expand=False)
    metadata["dialogue_id"] = metadata["file"].str.replace(r"_[^_]+\.wav$", "", regex=True)
    return metadata


def filter_iemocap_metadata(
    metadata,
    label_column="major_emotion",
    drop_labels=None,
    min_samples_per_class=30,
    max_samples=None,
    random_state=42,
):
    filtered = metadata.copy()

    if drop_labels:
        filtered = filtered[~filtered[label_column].isin(list(drop_labels))].copy()

    counts = filtered[label_column].value_counts()
    keep_labels = counts[counts >= min_samples_per_class].index
    filtered = filtered[filtered[label_column].isin(keep_labels)].copy()

    if max_samples is not None and len(filtered) > max_samples:
        keep_idx, _ = train_test_split(
            filtered.index.to_numpy(),
            train_size=max_samples,
            random_state=random_state,
            stratify=filtered[label_column].to_numpy(),
        )
        filtered = filtered.loc[np.sort(keep_idx)].copy()

    filtered = filtered.reset_index(drop=True)
    counts = filtered[label_column].value_counts().sort_values(ascending=False)
    return filtered, counts


def load_audio_from_bytes(audio_payload, target_sr=DEFAULT_SAMPLE_RATE):
    waveform, sample_rate = sf.read(io.BytesIO(audio_payload["bytes"]), dtype="float32")

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    if target_sr is not None and sample_rate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr

    return waveform.astype(np.float32), sample_rate


def extract_wave2vec_features_from_rows(
    dataset,
    row_indices,
    processor,
    wave2vec_model,
    device,
    cache_path=None,
    target_sr=DEFAULT_SAMPLE_RATE,
):
    row_indices = np.asarray(row_indices, dtype=np.int64)

    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        if np.array_equal(cached["row_indices"], row_indices):
            print(f"Loaded cached audio features from {cache_path}")
            return cached["X"].astype(np.float32)

    wave2vec_model.eval()
    features = []

    for row_idx in tqdm(row_indices, desc="Extracting audio features"):
        example = dataset[int(row_idx)]
        waveform, sample_rate = load_audio_from_bytes(example["audio"], target_sr=target_sr)

        inputs = processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            hidden_states = wave2vec_model(**inputs).last_hidden_state.squeeze(0)

        pooled = torch.cat(
            [
                torch.mean(hidden_states, dim=0),
                torch.std(hidden_states, dim=0),
                torch.max(hidden_states, dim=0)[0],
                torch.min(hidden_states, dim=0)[0],
            ]
        )
        features.append(pooled.cpu().numpy().astype(np.float32))

    X = np.asarray(features, dtype=np.float32)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, row_indices=row_indices, X=X)
        print(f"Saved audio features to {cache_path}")

    return X


def transcribe_rows_with_whisper(
    dataset,
    row_indices,
    whisper_model,
    cache_path=None,
    target_sr=DEFAULT_SAMPLE_RATE,
):
    row_indices = np.asarray(row_indices, dtype=np.int64)

    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        if np.array_equal(cached["row_indices"], row_indices):
            print(f"Loaded cached Whisper transcriptions from {cache_path}")
            return cached["texts"].tolist()

    transcripts = []
    use_fp16 = getattr(getattr(whisper_model, "device", torch.device("cpu")), "type", "cpu") == "cuda"

    for row_idx in tqdm(row_indices, desc="Transcribing with Whisper"):
        example = dataset[int(row_idx)]
        waveform, _ = load_audio_from_bytes(example["audio"], target_sr=target_sr)
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
            row_indices=row_indices,
            texts=np.asarray(transcripts, dtype=object),
        )
        print(f"Saved Whisper transcriptions to {cache_path}")

    return transcripts


def embed_texts_with_distilbert(
    texts,
    tokenizer,
    bert_model,
    device,
    cache_path=None,
    batch_size=16,
    max_length=128,
):
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        if len(cached["texts"]) == len(texts) and np.array_equal(cached["texts"], np.asarray(texts, dtype=object)):
            print(f"Loaded cached text embeddings from {cache_path}")
            return cached["X"].astype(np.float32)

    bert_model.eval()
    embeddings = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
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
            hidden_states = bert_model(**encoded).last_hidden_state

        attention_mask = encoded["attention_mask"].bool()

        for batch_idx in range(hidden_states.shape[0]):
            valid_tokens = hidden_states[batch_idx][attention_mask[batch_idx]]
            pooled = torch.cat(
                [
                    torch.mean(valid_tokens, dim=0),
                    torch.std(valid_tokens, dim=0),
                    torch.max(valid_tokens, dim=0)[0],
                    torch.min(valid_tokens, dim=0)[0],
                ]
            )
            embeddings.append(pooled.cpu().numpy().astype(np.float32))

    X = np.asarray(embeddings, dtype=np.float32)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            texts=np.asarray(texts, dtype=object),
            X=X,
        )
        print(f"Saved text embeddings to {cache_path}")

    return X


def build_label_maps(labels):
    label_names = sorted(pd.Series(labels).unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_names, label_to_idx, idx_to_label


def encode_labels(labels, label_to_idx):
    return np.asarray([label_to_idx[label] for label in labels], dtype=np.int64)


def session_holdout_split(metadata, test_sessions=("Ses05",)):
    test_mask = metadata["session_id"].isin(list(test_sessions)).to_numpy()
    train_mask = ~test_mask

    if not train_mask.any():
        raise ValueError("No training samples left after session holdout split.")
    if not test_mask.any():
        raise ValueError("No test samples selected by the session holdout split.")

    return train_mask, test_mask


def train_val_test_split(
    metadata,
    label_column="major_emotion",
    test_sessions=("Ses05",),
    validation_size=0.15,
    random_state=42,
):
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1.")

    train_eval_mask, test_mask = session_holdout_split(metadata, test_sessions=test_sessions)
    train_eval_idx = np.flatnonzero(train_eval_mask)
    train_eval_labels = metadata.iloc[train_eval_idx][label_column].to_numpy()

    train_idx, val_idx = train_test_split(
        train_eval_idx,
        test_size=validation_size,
        random_state=random_state,
        stratify=train_eval_labels,
    )

    train_mask = np.zeros(len(metadata), dtype=bool)
    val_mask = np.zeros(len(metadata), dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    if not train_mask.any():
        raise ValueError("No training samples left after validation split.")
    if not val_mask.any():
        raise ValueError("No validation samples selected by the validation split.")

    return train_mask, val_mask, test_mask


def scale_feature_splits(X_train, *other_splits):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    scaled_splits = [X_train_scaled.astype(np.float32)]

    for split in other_splits:
        scaled_splits.append(scaler.transform(split).astype(np.float32))

    return (*scaled_splits, scaler)


def scale_train_test(X_train, X_test):
    X_train_scaled, X_test_scaled, scaler = scale_feature_splits(X_train, X_test)
    return X_train_scaled, X_test_scaled, scaler


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class EmotionClassifier(nn.Module):
    def __init__(self, input_size, num_emotions, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_emotions)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, features):
        x = self.dropout(self.relu(self.bn1(self.fc1(features))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)


def create_feature_loader(features, labels, batch_size=32, shuffle=False):
    dataset = FeatureDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    train_loader = create_feature_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    test_loader = create_feature_loader(X_test, y_test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def create_train_val_test_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size=32,
):
    train_loader = create_feature_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = create_feature_loader(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = create_feature_loader(X_test, y_test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def compute_class_weights(y_train, num_classes):
    counts = np.bincount(y_train, minlength=num_classes)
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = len(y_train) / (num_classes * counts[nonzero])
    return torch.tensor(weights, dtype=torch.float32), counts


def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in data_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    return total_loss / max(len(data_loader), 1), 100.0 * correct / max(total, 1)


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / max(len(data_loader), 1),
        100.0 * correct / max(total, 1),
        np.asarray(all_preds),
        np.asarray(all_labels),
    )


def train_model(
    model,
    train_loader,
    test_loader,
    device,
    num_epochs=30,
    learning_rate=1e-3,
    class_weights=None,
    eval_name="test",
    early_stopping_patience=None,
    restore_best=False,
    monitor="loss",
):
    if eval_name not in {"test", "val"}:
        raise ValueError("eval_name must be either 'test' or 'val'.")
    if monitor not in {"loss", "acc"}:
        raise ValueError("monitor must be either 'loss' or 'acc'.")

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "train_acc": [],
        f"{eval_name}_loss": [],
        f"{eval_name}_acc": [],
    }
    best_state_dict = None
    best_epoch = None
    best_metric = None
    epochs_without_improvement = 0

    print(f"Training on {device}...")

    for epoch in tqdm(range(num_epochs), desc="Training"):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate_model(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history[f"{eval_name}_loss"].append(test_loss)
        history[f"{eval_name}_acc"].append(test_acc)

        current_metric = test_loss if monitor == "loss" else test_acc
        improved = False

        if best_metric is None:
            improved = True
        elif monitor == "loss" and current_metric < best_metric:
            improved = True
        elif monitor == "acc" and current_metric > best_metric:
            improved = True

        if improved:
            best_metric = current_metric
            best_epoch = epoch + 1
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
                f"{eval_name}_loss={test_loss:.4f} {eval_name}_acc={test_acc:.2f}%"
            )

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(best {eval_name}_{monitor} at epoch {best_epoch}: {best_metric:.4f})"
            )
            break

    if restore_best and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    history["best_epoch"] = best_epoch
    history[f"best_{eval_name}_{monitor}"] = best_metric
    return history
