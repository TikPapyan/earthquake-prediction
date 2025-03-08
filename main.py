from data_processing import process_earthquake_data
from feature_engineering import feature_strength_analysis, plot_earthquake_data

initial_data = "data/earthquake.csv"
active_faults_data = "data/gem_active_faults_harmonized.geojson"
processed_data = "data/earthquake_processed.csv"
correlation_matrix = 'correlation_matrix.png'
visualization = 'visualizations'

if __name__ == "__main__":
    process_earthquake_data(initial_data, active_faults_data, processed_data)
    feature_strength_analysis(processed_data, correlation_matrix, save_figures=True, save_dir=visualization)
    plot_earthquake_data(processed_data, save_figures=True, save_dir=visualization)