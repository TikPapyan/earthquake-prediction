import json
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from scipy.interpolate import interp1d

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
    
    df["mag"] = np.clip(df["mag"], 4.0, 5.6)
    df["depth"] = np.clip(df["depth"], 0, 67.5)
    
    scaler = MinMaxScaler()
    df[["depth", "mag", "Distance_to_Nearest_Fault", "TimeToNext"]] = scaler.fit_transform(
        df[["depth", "mag", "Distance_to_Nearest_Fault", "TimeToNext"]]
    )

def augment_earthquake_data(input_file, output_augmented, output_train, output_test, test_size=0.2, use_smote=False):
    df = pd.read_csv(input_file)

    for col in ["depth", "mag", "Distance_to_Nearest_Fault", "TimeToNext"]:
        interp_func = interp1d(np.arange(len(df)), df[col], kind="linear", fill_value="extrapolate")
        df[f"{col}_augmented"] = interp_func(np.arange(len(df)) + np.random.uniform(-0.5, 0.5, len(df)))

    if use_smote:
        smote = SMOTE(sampling_strategy="auto", random_state=42)
        features = ["depth_augmented", "Distance_to_Nearest_Fault_augmented", "TimeToNext_augmented"]
        target = "mag"
        X_resampled, y_resampled = smote.fit_resample(df[features], df[target])
        df = pd.DataFrame(X_resampled, columns=features)
        df[target] = y_resampled
    
    train_size = int((1 - test_size) * len(df))
    train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]
    
    df.to_csv(output_augmented, index=False)
    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)
    
    print(f"Augmented dataset saved as {output_augmented}")
    print(f"Train dataset saved as: {output_train}")
    print(f"Test dataset saved as: {output_test}")