# Configuration and Constants for Emotion Classification

# Device configuration
USE_CUDA = True

# Emotion dictionary (RAVDESS emotions)
EMOTION_DICT = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgusted',
    '08': 'surprised'
}

# Model configuration
MODEL_CONFIG = {
    'input_size': 3072,  # Wave2Vec2: 768 * 4 (mean, std, max, min)
    'hidden_sizes': [512, 256, 128],
    'num_emotions': 8,
    'dropout': 0.5,
    'batch_size': 16,
    'learning_rate': 0.001,
    'num_epochs': 30,
    'test_size': 0.2,
    'random_state': 42
}

# Wave2Vec2 configuration
WAVE2VEC2_MODEL = "facebook/wav2vec2-base"
WAVE2VEC2_SAMPLE_RATE = 16000
WAVE2VEC2_FEATURE_DIM = 768

# Dataset paths
RAVDESS_FOLDER = 'RAVDESS_data'
MODELS_FOLDER = 'trained_models'

# Feature aggregation (mean, std, max, min from 768 dimensions)
FEATURE_AGGREGATION_METHODS = ['mean', 'std', 'max', 'min']
