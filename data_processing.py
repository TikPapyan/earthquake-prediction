import json
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from sklearn.preprocessing import MinMaxScaler

def calculate_distance_to_nearest_fault(row, fault_lines):
    earthquake_point = Point(row["longitude"], row["latitude"])
    nearest_fault = min(fault_lines, key=lambda line: line.distance(earthquake_point))
    nearest_point_on_fault, _ = nearest_points(nearest_fault, earthquake_point)
    return nearest_point_on_fault.distance(earthquake_point)

def process_earthquake_data(earthquake_file, fault_geojson, output_preprocessed, output_train, output_test, test_size=0.2):
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
    
    # Train-Test Split (Time-based)
    train_size = int((1 - test_size) * len(df))
    train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]
    
    df.to_csv(output_preprocessed, index=False)
    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)
    
    print(f"Preprocessed dataset saved as {output_preprocessed}")
    print(f"Train dataset saved as: {output_train}")
    print(f"Test dataset saved as: {output_test}")
