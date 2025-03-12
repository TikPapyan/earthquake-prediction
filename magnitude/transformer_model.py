import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=8, num_layers=2):
        super(TransformerPredictor, self).__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x.permute(1, 0, 2))
        x = x.mean(dim=0)
        x = self.fc(x)
        return x

def train_transformer_model(train_data_path, test_data_path, seq_length=10, epochs=100, learning_rate=0.001):
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    features = ["depth", "rms", "magNst", "Distance_to_Nearest_Fault", "TimeToNext", "year", "month", "day", "hour", "depth_augmented", "Distance_to_Nearest_Fault_augmented", "TimeToNext_augmented"]
    target = "mag"

    def create_sequences(df, features, target, seq_length):
        X, y = [], []
        for i in range(len(df) - seq_length):
            X.append(df[features].iloc[i:i+seq_length].values)
            y.append(df[target].iloc[i+seq_length])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_df, features, target, seq_length)
    X_test, y_test = create_sequences(test_df, features, target, seq_length)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    model = TransformerPredictor(input_dim=X_train.shape[2])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        test_loss = criterion(predictions, y_test_tensor).item()
        rmse = torch.sqrt(torch.tensor(test_loss))
        r2 = r2_score(y_test, predictions.detach().numpy())
        print(f"Test Loss: {test_loss:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    return model
