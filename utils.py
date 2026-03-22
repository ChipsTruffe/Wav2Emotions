# Utility functions for data loading and feature extraction

import os
import numpy as np
import librosa
import torch
from config import EMOTION_DICT, WAVE2VEC2_SAMPLE_RATE, RAVDESS_FOLDER


def extract_wave2vec_features(audio_path, processor, wave2vec_model, device):
    """
    Extract Wave2Vec2 embeddings from audio file.
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file
    processor : Wav2Vec2Processor
        Wave2Vec2 processor for audio preprocessing
    wave2vec_model : Wav2Vec2Model
        Pre-trained Wave2Vec2 model
    device : torch.device
        Device to run inference on (cuda or cpu)
    
    Returns:
    --------
    features : np.ndarray or None
        3072-dimensional feature vector (768 * 4) or None if error
    """
    try:
        # Load audio using librosa (no torchcodec required)
        waveform_np, sample_rate = librosa.load(audio_path, sr=None, mono=False)
        
        # Ensure it's 2D (channels, samples) or convert to 1D
        if waveform_np.ndim == 1:
            waveform_np = np.expand_dims(waveform_np, axis=0)
        
        # Resample to 16kHz if necessary (Wave2Vec2 expects 16kHz)
        if sample_rate != WAVE2VEC2_SAMPLE_RATE:
            waveform_np = librosa.resample(waveform_np, orig_sr=sample_rate, target_sr=WAVE2VEC2_SAMPLE_RATE)
        
        # Convert to mono if stereo
        if waveform_np.shape[0] > 1:
            waveform_np = np.mean(waveform_np, axis=0, keepdims=True)
        
        # Convert to torch tensor
        waveform = torch.FloatTensor([waveform_np[0]])  # Remove batch dimension for processing
        
        # Process audio
        inputs = processor(waveform.squeeze(), sampling_rate=WAVE2VEC2_SAMPLE_RATE, return_tensors="pt", padding=True)
        
        # Extract embeddings
        with torch.no_grad():
            outputs = wave2vec_model(**inputs.to(device))
            hidden_states = outputs.last_hidden_state  # Shape: (1, time_steps, 768)
        
        # Aggregate temporal dimension using statistics
        embeddings = hidden_states.squeeze(0)  # (time_steps, 768)
        
        # Calculate statistics across time dimension
        mean_emb = torch.mean(embeddings, dim=0)  # (768,)
        std_emb = torch.std(embeddings, dim=0)   # (768,)
        max_emb = torch.max(embeddings, dim=0)[0] # (768,)
        min_emb = torch.min(embeddings, dim=0)[0] # (768,)
        
        # Concatenate features (768 * 4 = 3072 features)
        features = torch.cat([mean_emb, std_emb, max_emb, min_emb]).cpu().numpy()
        
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


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
