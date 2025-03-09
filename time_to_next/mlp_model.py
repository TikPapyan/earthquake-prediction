import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

class OptimizedMLP(nn.Module):
    def __init__(self, input_dim, hidden_size=128, dropout_rate=0.3):
        super(OptimizedMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        
def train_optimized_mlp(train_data_path, test_data_path, epochs=100, learning_rate=0.001, patience=10):
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    features = ["depth", "mag", "Distance_to_Nearest_Fault", "year", "month", "day", "hour"]
    target = "TimeToNext"

    X_train, y_train = train_df[features].values, train_df[target].values
    X_test, y_test = test_df[features].values, test_df[target].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    model = OptimizedMLP(input_dim=X_train.shape[1], hidden_size=128, dropout_rate=0.3)

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
    
    plt.scatter(y_test, predictions.detach().numpy())
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.show()

    # torch.save(model.state_dict(), "optimized_mlp_time_to_next.pth")
    # print("Optimized MLP Model Saved!")

    return model, scaler
