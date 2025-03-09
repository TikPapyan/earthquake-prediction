from data_processing import process_earthquake_data
from feature_engineering import feature_strength_analysis, plot_earthquake_data
from time_to_next.mlp_model import train_optimized_mlp
# from lstm_model import train_optimized_lstm
from magnitude.lstm_model import random_search_lstm
from magnitude.transformer_model import train_transformer_model

initial_data = "data/earthquake_data.csv"
active_faults_data = "data/fault_lines.geojson"
processed_data = "data/earthquake_data_preprocessed.csv"
train_data = "data/train_data.csv"
test_data = "data/test_data.csv"
correlation_matrix = 'correlation_matrix.png'
visualization = 'visualizations'

if __name__ == "__main__":
    # process_earthquake_data(
    #     initial_data,
    #     active_faults_data,
    #     processed_data,
    #     train_data,
    #     test_data
    # )
    # feature_strength_analysis(processed_data, correlation_matrix, save_figures=True, save_dir=visualization)
    # plot_earthquake_data(processed_data, save_figures=True, save_dir=visualization)
    # model, scaler = train_optimized_mlp(train_data, test_data)
    # model, scaler = train_optimized_lstm(train_data, test_data)
    model, scaler = random_search_lstm(train_data, test_data)
    model = train_transformer_model(train_data, test_data)
