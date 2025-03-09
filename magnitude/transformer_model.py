import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, nhead=8, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_layers = num_layers
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.nhead, dropout=dropout),
            num_layers=self.num_layers
        )
        
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.permute(1, 0, 2)
        output = self.encoder(x)
        output = output.mean(dim=0)
        output = self.fc2(output)
        return output

def train_transformer_model(train_data_path, test_data_path, epochs=100, learning_rate=0.001):
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    features = ["depth", "rms", "magNst", "Distance_to_Nearest_Fault", "TimeToNext", "year", "month", "day", "hour", "depth_augmented", "Distance_to_Nearest_Fault_augmented", "TimeToNext_augmented"]
    target = "mag"

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

    model = TransformerModel(input_dim=X_train.shape[2])
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
