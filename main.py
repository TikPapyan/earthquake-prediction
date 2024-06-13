from sklearn.model_selection import train_test_split
from rf_model import load_data, train_and_evaluate

if __name__ == "__main__":
    processed_file = 'data/earthquake_processed.csv'
    X, y = load_data(processed_file)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model = train_and_evaluate(X_train, X_test, y_train, y_test)