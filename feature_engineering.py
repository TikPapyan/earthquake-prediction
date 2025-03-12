import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def feature_engineering(input_file: str, output_file: str, missing_threshold: float = 0.3):
    df = pd.read_csv(input_file)

    df.dropna(thresh=len(df) * (1 - missing_threshold), axis=1, inplace=True)

    df['rms'].fillna(df['rms'].mean(), inplace=True)
    df['magNst'].fillna(df['magNst'].median(), inplace=True)

    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', inplace=True)
    df['TimeToNext'] = df['time'].diff().dt.total_seconds().div(86400)
    df['TimeToNext'].fillna(method='bfill', inplace=True)

    df['Rolling_Mag_30D'] = df['mag'].rolling(window=30, min_periods=1).mean()

    df['Depth_Category'] = pd.cut(df['depth'], bins=[0, 70, 300, 700], labels=['shallow', 'intermediate', 'deep'])

    df.to_csv(output_file, index=False)
    print(f"Processed file saved as: {output_file}")

def feature_strength_analysis(processed_file, output_image, save_figures=False, save_dir="visualizations"):
    df = pd.read_csv(processed_file)
    df.drop(columns=['time'], inplace=True, errors='ignore')

    correlation_matrix = df.corr()
    feature_correlations = correlation_matrix['mag'].sort_values(ascending=False)

    print("Correlation of each feature with 'mag':")
    print(feature_correlations)

    if save_figures:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        output_path = os.path.join(save_dir, output_image)
        if os.path.exists(output_path):
            print(f"Warning: Overwriting existing file {output_path}")

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved as {output_path}")

    return feature_correlations

def plot_earthquake_data(csv_file_path: str, save_figures: bool = False, save_dir: str = "visualizations"):
    if save_figures and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    df = pd.read_csv(csv_file_path)
    df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)

    features = ['mag', 'depth', 'Distance_to_Nearest_Fault', 'TimeToNext']
    
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(features, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
    
    plt.tight_layout()
    if save_figures:
        plt.savefig(os.path.join(save_dir, 'distribution_plots.png'), dpi=300, bbox_inches='tight')
        print("Distribution plots saved.")
    else:
        plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['depth'], y=df['mag'], alpha=0.6)
    plt.xlabel('Depth (km)')
    plt.ylabel('Magnitude')
    plt.title('Magnitude vs Depth')
    
    if save_figures:
        plt.savefig(os.path.join(save_dir, 'mag_vs_depth.png'), dpi=300, bbox_inches='tight')
        print("Magnitude vs Depth plot saved.")
    else:
        plt.show()

    plt.figure(figsize=(10, 5))
    df['year'] = df['time'].dt.year
    df['year'].value_counts().sort_index().plot(kind='line')
    plt.xlabel('Year')
    plt.ylabel('Number of Earthquakes')
    plt.title('Earthquake Frequency Over Years')
    
    if save_figures:
        plt.savefig(os.path.join(save_dir, 'earthquake_frequency_over_years.png'), dpi=300, bbox_inches='tight')
        print("Earthquake frequency plot saved.")
    else:
        plt.show()

    plt.figure(figsize=(10, 5))
    df['month'] = df['time'].dt.month
    df.groupby('month')['mag'].count().plot(kind='bar')
    plt.xlabel('Month')
    plt.ylabel('Number of Earthquakes')
    plt.title('Earthquakes by Month')
    
    if save_figures:
        plt.savefig(os.path.join(save_dir, 'earthquakes_by_month.png'), dpi=300, bbox_inches='tight')
        print("Earthquakes by month plot saved.")
    else:
        plt.show()

    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df['mag'])
    plt.title('Magnitude Outliers')
    
    if save_figures:
        plt.savefig(os.path.join(save_dir, 'magnitude_outliers.png'), dpi=300, bbox_inches='tight')
        print("Magnitude outliers plot saved.")
    else:
        plt.show()

    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df['depth'])
    plt.title('Depth Outliers')
    
    if save_figures:
        plt.savefig(os.path.join(save_dir, 'depth_outliers.png'), dpi=300, bbox_inches='tight')
        print("Depth outliers plot saved.")
    else:
        plt.show()

def load_data(processed_file: str):
    df = pd.read_csv(processed_file)
    X = df.drop(columns=['time', 'longitude', 'latitude'])
    y = df['mag']
    return X, y