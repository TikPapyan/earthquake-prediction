import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def feature_strength_analysis(processed_file):
    df = pd.read_csv(processed_file)
    
    if 'time' in df.columns:
        df.drop(columns=['time'], inplace=True)
    
    correlation_matrix = df.corr()
    
    feature_correlations = correlation_matrix['mag'].sort_values(ascending=False)
    
    print("Correlation of each feature with 'mag':")
    print(feature_correlations)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()

    return feature_correlations

if __name__ == "__main__":
    processed_file = 'data/earthquake_processed.csv'
    feature_correlations = feature_strength_analysis(processed_file)
