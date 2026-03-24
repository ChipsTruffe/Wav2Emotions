# Utility functions for data loading and feature extraction

import os
import numpy as np
import librosa
import torch
import pandas as pd
from config import EMOTION_DICT, WAVE2VEC2_SAMPLE_RATE, RAVDESS_FOLDER, MELD_FOLDER, MELD_EMOTION_LIST, MELD_EMOTIONS

def extract_wave2vec_features(audio_path, processor, wave2vec_model, device):
    
        # Direct load handles resampling and mono conversion
        # Use the constant WAVE2VEC2_SAMPLE_RATE defined globally
        waveform_np, _ = librosa.load(audio_path, sr=WAVE2VEC2_SAMPLE_RATE, mono=True)
        
        # Prepare inputs for the model
        inputs = processor(
            waveform_np, 
            sampling_rate=WAVE2VEC2_SAMPLE_RATE, 
            return_tensors="pt", 
            padding=True
        ).to(device)
        
        # Extract hidden states
        with torch.no_grad():
            outputs = wave2vec_model(**inputs)
            # Shape: (1, time_steps, 768)
            embeddings = outputs.last_hidden_state.squeeze(0) 
        
        # Calculate statistics across the temporal dimension (dim=0)
        mean_emb = torch.mean(embeddings, dim=0)
        std_emb = torch.std(embeddings, dim=0)
        max_emb = torch.max(embeddings, dim=0)[0]
        min_emb = torch.min(embeddings, dim=0)[0]
        
        # Concatenate into a 3072-dimensional vector
        features = torch.cat([mean_emb, std_emb, max_emb, min_emb]).cpu().numpy()
        
        return features


def load_ravdess_data(data_folder=RAVDESS_FOLDER, processor=None, wave2vec_model=None, device=None):
    """
    Load audio files and emotions from RAVDESS dataset folder.
    
    RAVDESS filename format: MM-VC-EM-IN-ST-RE-AC.wav
    - MM: Modality (01=speech, 02=song)
    - VC: Vocal channel (01=speech, 02=song)
    - EM: Emotion code (01=neutral, ..., 08=surprised)
    - IN: Intensity (01=normal, 02=strong)
    - ST: Statement (01 or 02)
    - RE: Repetition (01 or 02)
    - AC: Actor (01-24)
    
    Example: 03-01-06-02-02-02-24.wav = speech/speech/fearful/strong/st1/rep2/actor24
    
    Parameters:
    -----------
    data_folder : str
        Path to RAVDESS dataset folder
    processor : Wav2Vec2Processor
        Wave2Vec2 processor for preprocessing
    wave2vec_model : Wav2Vec2Model
        Pre-trained Wave2Vec2 model
    device : torch.device
        Device to run inference on
    
    Returns:
    --------
    X, y : tuple of lists
        Feature vectors (X) and emotion labels (y)
    """
    X = []
    y = []
    features_list = []
    emotions_list = []
    
    # Walk through RAVDESS folder structure
    if os.path.exists(data_folder):
        file_count = 0
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.endswith('.wav'):
                    try:
                        # Extract emotion code from filename
                        # Format: MM-VC-EM-... where EM is emotion code (01-08) at position [2]
                        parts = file.split('-')
                        if len(parts) >= 3:
                            emotion_code = parts[2]
                            if emotion_code in EMOTION_DICT:
                                audio_path = os.path.join(root, file)
                                
                                # Extract features
                                features = extract_wave2vec_features(audio_path, processor, wave2vec_model, device)
                                if features is not None:
                                    features_list.append(features)
                                    emotions_list.append(EMOTION_DICT[emotion_code])
                                    file_count += 1
                    except Exception as e:
                        print(f"  Error processing {file}: {e}")
        
        if len(features_list) > 0:
            X.extend(features_list)
            y.extend(emotions_list)
            print(f"Loaded {len(features_list)} audio files")
        else:
            print(f"No audio files found in {data_folder}")
    else:
        print(f"Folder not found: {data_folder}")
    
    return X, y


def check_emotion_distribution(y):
    """
    Check emotion distribution and validate dataset.
    
    Parameters:
    -----------
    y : list
        List of emotion labels
    """
    if len(y) > 0:
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) == 1:
            print(f"Warning: Single emotion class found")
            return False
        else:
            print(f"Found {len(unique)} emotion classes in {len(y)} samples")
            return True
    else:
        print("No data loaded")
        return False


from tqdm import tqdm

def load_meld_data(data_folder=MELD_FOLDER, processor=None, wave2vec_model=None, device=None, 
                   split='train', max_samples=None, cache_path=None,
                   use_cache=True, save_cache=True):
    """
    Load MELD data and optionally cache extracted wav2vec embeddings.

    Parameters:
    -----------
    split : str
        MELD split name (train/dev/test)
    max_samples : int or None
        Optional cap on number of processed rows
    cache_path : str or None
        Optional path to a .npz cache file for embeddings and labels
    use_cache : bool
        If True and cache_path exists, load cached embeddings instead of recomputing
    save_cache : bool
        If True and cache_path is set, save extracted embeddings after processing
    """
    X, y = [], []

    if cache_path and use_cache and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        X_cached = cached["X"]
        y_cached = cached["y"]
        X = [np.asarray(row, dtype=np.float32) for row in X_cached]
        y = y_cached.tolist()
        print(f"Loaded {len(X)} cached samples from {cache_path}")
        return X, y
    
    csv_file = os.path.join(data_folder, f'{split}.csv')
    audio_folder = os.path.join(data_folder, split)
    
    if not os.path.exists(csv_file) or not os.path.exists(audio_folder):
        print(f"Missing data path for {split}")
        print(f"  Expected CSV: {csv_file}")
        print(f"  Expected audio folder: {audio_folder}")
        return X, y
    
    # Use latin-1 to avoid UnicodeDecodeErrors common in MELD
    df = pd.read_csv(csv_file, encoding='latin-1')
    
    # Ensure all list elements are lowercase for robust matching
    valid_emotions = [e.lower() for e in MELD_EMOTION_LIST]
    
    loaded_count = 0
    # Added tqdm for progress tracking during inference
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
        if max_samples and loaded_count >= max_samples:
            break
        
        emotion = str(row['Emotion']).strip().lower()
        if emotion not in valid_emotions:
            continue
        
        # Ensure integer IDs to avoid 'dia1.0_utt0.0.flac' errors
        dia_id = int(row['Dialogue_ID'])
        utt_id = int(row['Utterance_ID'])
        audio_file = os.path.join(audio_folder, f"dia{dia_id}_utt{utt_id}.flac")
        
        if not os.path.exists(audio_file):
            continue
        
        features = extract_wave2vec_features(audio_file, processor, wave2vec_model, device)
        if features is not None:
            X.append(features)
            # Map MELD emotions to RAVDESS emotions
            ravdess_emotion = MELD_EMOTIONS.get(emotion, emotion)
            y.append(ravdess_emotion)
            loaded_count += 1

    if cache_path and save_cache and len(X) > 0:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(
            cache_path,
            X=np.asarray(X, dtype=np.float32),
            y=np.asarray(y, dtype=object)
        )
        print(f"Saved {len(X)} samples to cache: {cache_path}")

    print(f"Successfully loaded {len(X)} samples.")
    return X, y