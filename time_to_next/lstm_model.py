import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

class OptimizedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size=128, dropout_rate=0.3, num_layers=1):
        super(OptimizedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = out[:, -1, :]
        
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out

def train_optimized_lstm(train_data_path, test_data_path, epochs=100, learning_rate=0.001, patience=10):
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    features = ["depth", "mag", "Distance_to_Nearest_Fault", "year", "month", "day", "hour"]
    target = "TimeToNext"

    X_train, y_train = train_df[features].values, train_df[target].values
    X_test, y_test = test_df[features].values, test_df[target].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    model = OptimizedLSTM(input_dim=X_train.shape[2], hidden_size=128, dropout_rate=0.3, num_layers=1)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor).item()

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}")

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        test_loss = criterion(predictions, y_test_tensor).item()
        print(f"Final Test MSE Loss: {test_loss:.4f}")

        rmse = torch.sqrt(torch.tensor(test_loss))
        print(f"Final Test RMSE: {rmse:.4f}")

        mae = torch.mean(torch.abs(predictions - y_test_tensor))
        print(f"Test MAE: {mae:.4f}")

        r2 = r2_score(y_test, predictions.detach().numpy())
        print(f"R²: {r2:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions.detach().numpy(), color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title('Predicted vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.show()

    # torch.save(model.state_dict(), "optimized_lstm_time_to_next.pth")
    # print("Optimized LSTM Model Saved!")

    return model, scaler
