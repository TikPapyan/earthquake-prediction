import json
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

def calculate_distance_to_nearest_fault(row, fault_lines):
    earthquake_point = Point(row["longitude"], row["latitude"])
    nearest_fault = min(fault_lines, key=lambda line: line.distance(earthquake_point))
    nearest_point_on_fault, _ = nearest_points(nearest_fault, earthquake_point)
    return nearest_point_on_fault.distance(earthquake_point)

def process_earthquake_data(earthquake_file, fault_geojson, output_file):
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
    
    df.to_csv(output_file, index=False)
    print(f"Processed file saved as: {output_file}")
