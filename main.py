import argparse
from data_processing import process_earthquake_data, augment_earthquake_data
from feature_engineering import feature_strength_analysis, plot_earthquake_data
from time_to_next.lstm_model import train_optimized_lstm
from magnitude.transformer_model import train_transformer_model

def main():
    parser = argparse.ArgumentParser(description="Earthquake Prediction Pipeline")
    parser.add_argument("--process-data", action="store_true", help="Run data processing")
    parser.add_argument("--train-lstm", action="store_true", help="Train LSTM model")
    parser.add_argument("--train-transformer", action="store_true", help="Train Transformer model")
    args = parser.parse_args()

    initial_data = "data/earthquake_data.csv"
    active_faults_data = "data/fault_lines.geojson"
    processed_data = "data/earthquake_data_preprocessed.csv"
    augmented_data = 'data/augmented_earthquake_data.csv'
    train_data = "data/train_data.csv"
    test_data = "data/test_data.csv"

    if args.process_data:
        process_earthquake_data(initial_data, active_faults_data, processed_data, train_data, test_data)
        augment_earthquake_data(processed_data, augmented_data, train_data, test_data, use_smote=False)
        feature_strength_analysis(augmented_data, "correlation_matrix.png", save_figures=True, save_dir="visualizations")
        plot_earthquake_data(processed_data, save_figures=True, save_dir="visualizations")

    if args.train_lstm:
        model, scaler = train_optimized_lstm(train_data, test_data)

    if args.train_transformer:
        model = train_transformer_model(train_data, test_data)

if __name__ == "__main__":
    main()
