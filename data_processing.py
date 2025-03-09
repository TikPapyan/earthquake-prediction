import json
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

def calculate_distance_to_nearest_fault(row, fault_lines):
    earthquake_point = Point(row["longitude"], row["latitude"])
    nearest_fault = min(fault_lines, key=lambda line: line.distance(earthquake_point))
    nearest_point_on_fault, _ = nearest_points(nearest_fault, earthquake_point)
    return nearest_point_on_fault.distance(earthquake_point)

def process_earthquake_data(earthquake_file, fault_geojson):
    df = pd.read_csv(earthquake_file, sep=';')
    
    with open(fault_geojson, 'r') as f:
        fault_data = json.load(f)
    fault_lines = [LineString(feature["geometry"]["coordinates"]) for feature in fault_data["features"]]
    
    df["Distance_to_Nearest_Fault"] = df.apply(calculate_distance_to_nearest_fault, fault_lines=fault_lines, axis=1)
    
    missing_threshold = 0.3
    df.dropna(thresh=len(df) * (1 - missing_threshold), axis=1, inplace=True)
    df["rms"].fillna(df["rms"].mean(), inplace=True)
    df["magNst"].fillna(df["magNst"].median(), inplace=True)
    
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.sort_values(by="time", inplace=True)
    df["TimeToNext"] = df["time"].diff().dt.total_seconds().div(86400).fillna(0)
    
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["hour"] = df["time"].dt.hour
    
    # Handling Outliers
    df["mag"] = np.clip(df["mag"], 4.0, 5.6)
    df["depth"] = np.clip(df["depth"], 0, 67.5)
    
    # Feature Scaling
    scaler = MinMaxScaler()
    df[["depth", "mag", "Distance_to_Nearest_Fault", "TimeToNext"]] = scaler.fit_transform(
        df[["depth", "mag", "Distance_to_Nearest_Fault", "TimeToNext"]]
    )

def augment_earthquake_data(input_file, output_augmented, output_train, output_test, test_size=0.2, noise_factor=0.02, smote=False):
    df = pd.read_csv(input_file)
    
    for col in ["depth", "mag", "Distance_to_Nearest_Fault", "TimeToNext"]:
        noise = np.random.normal(0, df[col].std() * noise_factor, size=df.shape[0])
        df[f"{col}_augmented"] = df[col] + noise
    
    if smote:
        features = ["depth_augmented", "Distance_to_Nearest_Fault_augmented", "TimeToNext_augmented"]
        target = "mag"
        
        smote = SMOTE(sampling_strategy="auto", random_state=42)
        X_resampled, y_resampled = smote.fit_resample(df[features], df[target])
        
        df = pd.DataFrame(X_resampled, columns=features)
        df[target] = y_resampled

    # Train-Test Split (Time-based)
    train_size = int((1 - test_size) * len(df))
    train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]
    
    df.to_csv(output_augmented, index=False)
    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)
    
    print(f"Augmented dataset saved as {output_augmented}")
    print(f"Train dataset saved as: {output_train}")
    print(f"Test dataset saved as: {output_test}")
    print(f"Augmented dataset saved as {output_augmented}")