import random
import torch.optim as optim
import pandas as pd
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class OptimizedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size=128, dropout_rate=0.3, num_layers=1):
        super(OptimizedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output one value (magnitude)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = out[:, -1, :]  # Get the last output from the LSTM
        
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out

# Hyperparameter tuning function using random search
def random_search_lstm(train_data_path, test_data_path, num_trials=10):
    # Hyperparameter ranges
    learning_rates = [0.001, 0.0005, 0.0001]
    hidden_sizes = [64, 128, 256]
    batch_sizes = [16, 32, 64]
    num_epochs = [50, 100, 150]
    dropout_rates = [0.2, 0.3, 0.5]
    
    best_model = None
    best_r2 = -float("inf")
    best_params = {}

    # Read in data
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    features = ["depth", "rms", "magNst", "Distance_to_Nearest_Fault", "year", "month", "day", "hour"]
    target = "mag"  # Magnitude prediction

    X_train, y_train = train_df[features].values, train_df[target].values
    X_test, y_test = test_df[features].values, test_df[target].values
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Random search
    for trial in range(num_trials):
        # Randomly select hyperparameters
        lr = random.choice(learning_rates)
        hidden_size = random.choice(hidden_sizes)
        batch_size = random.choice(batch_sizes)
        num_epoch = random.choice(num_epochs)
        dropout_rate = random.choice(dropout_rates)
        
        print(f"Trial {trial+1}: Learning Rate = {lr}, Hidden Size = {hidden_size}, Batch Size = {batch_size}, Epochs = {num_epoch}, Dropout = {dropout_rate}")

        model = OptimizedLSTM(input_dim=X_train.shape[2], hidden_size=hidden_size, dropout_rate=dropout_rate, num_layers=1)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Train the model
        best_loss = float("inf")
        counter = 0

        for epoch in range(num_epoch):
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
                if counter >= 10:  # Early stopping
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor)
            r2 = r2_score(y_test, predictions.detach().numpy())
            print(f"R² for trial {trial+1}: {r2:.4f}")

            # Track the best model
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_params = {
                    "learning_rate": lr,
                    "hidden_size": hidden_size,
                    "batch_size": batch_size,
                    "num_epoch": num_epoch,
                    "dropout_rate": dropout_rate
                }

    print("Best R²: ", best_r2)
    print("Best Hyperparameters: ", best_params)

    return best_model, best_params
