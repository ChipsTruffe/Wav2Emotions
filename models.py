# Neural Network Models and Training Functions for Emotion Classification

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import MODEL_CONFIG, EMOTION_DICT


# Create emotion indexing from config
EMOTION_LABELS = list(EMOTION_DICT.values())
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTION_LABELS)}
IDX_TO_EMOTION = {idx: emotion for emotion, idx in EMOTION_TO_IDX.items()}


class EmotionDataset(Dataset):
    """PyTorch Dataset for emotion classification"""
    
    def __init__(self, features, labels):
        """
        Parameters:
        -----------
        features : np.ndarray
            Feature vectors (N, 3072)
        labels : list or np.ndarray
            Emotion labels
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor([EMOTION_TO_IDX[label] for label in labels])
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class EmotionClassifier(nn.Module):
    """Neural Network for Emotion Classification
    
    Architecture:
    - Input: 3072 (Wave2Vec2 features)
    - Hidden: 512 -> 256 -> 128 with batch norm and dropout
    - Output: 8 (emotion classes)
    - Dropout: 0.5 after each hidden layer (active only in training mode)
    """
    
    def __init__(self, input_size, num_emotions=8, dropout_rate=0.5):
        super(EmotionClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_emotions)
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(128)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x


def prepare_data(X, y, test_size=None, random_state=None):
    """
    Prepare data for training: split and normalize.
    
    This function performs train/test split BEFORE normalization to prevent data leakage.
    
    Parameters:
    -----------
    X : list or np.ndarray
        Feature vectors
    y : list or np.ndarray
        Labels
    test_size : float, optional
        Test set fraction
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    X_train, X_test, y_train, y_test, scaler
        Normalized train/test splits and fitted scaler
    """
    if test_size is None:
        test_size = MODEL_CONFIG['test_size']
    if random_state is None:
        random_state = MODEL_CONFIG['random_state']
    
    # Convert X to numpy array if needed
    X_array = np.array(X) if isinstance(X, list) else X
    
    # Encode labels
    y_encoded = np.array([EMOTION_TO_IDX[label] if isinstance(label, str) else label for label in y])
    
    # Split BEFORE normalization
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_array, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Fit scaler on training data ONLY
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    print(f"\nData normalization complete. Scaler fit on training data only.")
    
    return X_train, X_test, y_train, y_test, scaler


def create_dataloaders(X_train, X_test, y_train, y_test, batch_size=None):
    """
    Create PyTorch DataLoaders.
    
    Parameters:
    -----------
    X_train, X_test : np.ndarray
        Training and test features
    y_train, y_test : np.ndarray
        Training and test labels (encoded indices)
    batch_size : int, optional
        Batch size for dataloader
    
    Returns:
    --------
    train_loader, test_loader
        PyTorch DataLoaders
    """
    if batch_size is None:
        batch_size = MODEL_CONFIG['batch_size']
    
    train_dataset = EmotionDataset(X_train, [IDX_TO_EMOTION[idx] for idx in y_train])
    test_dataset = EmotionDataset(X_test, [IDX_TO_EMOTION[idx] for idx in y_test])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy, all_preds, all_labels


def train_model(model, train_loader, test_loader, device, num_epochs=None, learning_rate=None):
    """
    Train the emotion classifier model.
    
    Parameters:
    -----------
    model : EmotionClassifier
        Neural network model
    train_loader, test_loader : DataLoader
        Training and test data loaders
    device : torch.device
        Device to train on
    num_epochs : int, optional
        Number of training epochs
    learning_rate : float, optional
        Learning rate for optimizer
    
    Returns:
    --------
    train_losses, train_accs, test_losses, test_accs
        Training history
    """
    if num_epochs is None:
        num_epochs = MODEL_CONFIG['num_epochs']
    if learning_rate is None:
        learning_rate = MODEL_CONFIG['learning_rate']
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    print(f"Training Emotion Classifier on {device}...\n")
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%\n")
    
    print("Training complete!")
    
    return train_losses, train_accs, test_losses, test_accs


def analyze_training_quality(y_train, train_losses, test_losses):
    """
    Analyze training quality and potential issues.
    
    Parameters:
    -----------
    y_train : list or np.ndarray
        Training labels
    train_losses : list
        Training losses per epoch
    test_losses : list
        Test losses per epoch
    """
    # Check for single class
    unique_classes = len(set(y_train))
    if unique_classes == 1:
        print(f"Warning: Single class detected in training data. Model accuracy may not be meaningful.")
        return False
    
    # Check for overfitting
    if train_losses and test_losses:
        initial_train_loss = train_losses[0]
        final_train_loss = train_losses[-1]
        initial_test_loss = test_losses[0]
        final_test_loss = test_losses[-1]
        
        if initial_train_loss > 0:
            loss_reduction = (initial_train_loss - final_train_loss) / initial_train_loss * 100
            print(f"\nTrain loss improvement: {loss_reduction:.1f}%")
            
            if initial_test_loss > 0:
                test_loss_change = ((final_test_loss - initial_test_loss) / initial_test_loss * 100)
                if test_loss_change > 50:
                    print(f"Warning: Test loss increased by {test_loss_change:.1f}% - potential overfitting")
                else:
                    print(f"Test loss stable (change: {test_loss_change:.1f}%)")
    
    return True
