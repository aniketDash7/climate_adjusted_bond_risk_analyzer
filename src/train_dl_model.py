import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

class FirePathNet(nn.Module):
    """
    1D CNN + LSTM for Fire Path Prediction
    Input: (Batch, Sequence Length, Features)
    - Sequence Length: e.g., 7 days of historical weather
    - Features: Temp, Wind, Humidity, Precip, NDVI
    """
    def __init__(self, num_features=5, hidden_size=32, num_layers=2):
        super(FirePathNet, self).__init__()
        
        # 1D CNN to extract local temporal patterns. 
        # Expects input of shape: (Batch, Features, Sequence Length)
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # LSTM to capture the temporal progression
        # Expects input: (Batch, Sequence Length, Features)
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Fully connected layer for the final 48h forward probability
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, SeqLen, Features)
        # CNN expects (B, Features, SeqLen), so permute
        x = x.permute(0, 2, 1)
        
        # CNN block
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        # Permute back for LSTM: (B, SeqLen, Channels)
        x = x.permute(0, 2, 1)
        
        # LSTM block
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Take the output of the last time step
        last_out = lstm_out[:, -1, :]
        
        # FC Block
        out = self.fc(last_out)
        return self.sigmoid(out)

def load_real_sequences():
    """
    Load the real historical 7-day weather sequences fetched via OpenMeteo.
    X_real.npy: (N_samples, 7_days, 5_features)
    y_real.npy: (N_samples, 1)
    """
    print(f"Loading real weather sequences from {DATA_DIR}...")
    X_path = os.path.join(DATA_DIR, "X_real.npy")
    y_path = os.path.join(DATA_DIR, "y_real.npy")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Real dataset not found. Run generate_dl_dataset.py first.")
        
    X = np.load(X_path)
    y = np.load(y_path)
    
    # Simple standardization
    feature_means = X.mean(axis=(0,1))
    feature_stds = X.std(axis=(0,1)) + 1e-8
    
    X_norm = (X - feature_means) / feature_stds
    
    # Save scalers for inference
    joblib.dump({'means': feature_means, 'stds': feature_stds}, os.path.join(MODELS_DIR, 'dl_scaler.joblib'))
    
    return torch.tensor(X_norm, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def main():
    print("="*60)
    print("Training CNN+LSTM for Fire Path Prediction")
    print("="*60)
    
    X, y = load_real_sequences()
    
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = FirePathNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    print("Starting Training...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        val_loss /= len(val_loader.dataset)
        accuracy = correct / total
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {accuracy:.4f}")
        
    model_path = os.path.join(MODELS_DIR, "fire_path_cnn_lstm.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()
