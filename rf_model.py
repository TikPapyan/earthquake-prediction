import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data(processed_file):
    df = pd.read_csv(processed_file)
    X = df.drop(columns=['time', 'longitude', 'latitude', 'year', 'day', 'month', 'hour'])
    y = df['mag']
    return X, y

def train_and_evaluate(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestRegressor()

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")

    best_rf = grid_search.best_estimator_
    best_rf.fit(X_train_scaled, y_train)

    y_pred_train = best_rf.predict(X_train_scaled)
    y_pred_test = best_rf.predict(X_test_scaled)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")
    print(f"Train R2: {train_r2}")
    print(f"Test R2: {test_r2}")

    plot_evaluation_metrics(y_train, y_test, y_pred_train, y_pred_test, train_mse, test_mse, train_r2, test_r2)

    return best_rf

def plot_evaluation_metrics(y_train, y_test, y_pred_train, y_pred_test, train_mse, test_mse, train_r2, test_r2):
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_pred_train, color='blue', alpha=0.5)
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], linestyle='--', color='red')
    plt.xlabel('Actual Magnitude')
    plt.ylabel('Predicted Magnitude')
    plt.title('Actual vs Predicted (Training Set)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
    plt.xlabel('Actual Magnitude')
    plt.ylabel('Predicted Magnitude')
    plt.title('Actual vs Predicted (Test Set)')
    plt.grid(True)

    plt.tight_layout()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(['Train', 'Test'], [train_mse, test_mse], color=['blue', 'green'])
    plt.xlabel('Dataset')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error (MSE)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.bar(['Train', 'Test'], [train_r2, test_r2], color=['blue', 'green'])
    plt.xlabel('Dataset')
    plt.ylabel('R2 Score')
    plt.title('R2 Score')
    plt.grid(True)

    plt.tight_layout()
    plt.show()