import json
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

def calculate_distance_to_nearest_fault(earthquake_row, fault_lines):
    earthquake_point = Point(earthquake_row['longitude'], earthquake_row['latitude'])
    nearest_fault = min(fault_lines, key=lambda line: line.distance(earthquake_point))
    nearest_point_on_fault, _ = nearest_points(nearest_fault, earthquake_point)
    distance_to_fault = nearest_point_on_fault.distance(earthquake_point)
    return distance_to_fault

def feature_engineering(input_file, output_file, fault_geojson):
    df = pd.read_csv(input_file, sep=';')

    with open(fault_geojson, 'r') as f:
        fault_data = json.load(f)

    fault_lines = []
    for feature in fault_data['features']:
        coordinates = feature['geometry']['coordinates']
        line = LineString(coordinates)
        fault_lines.append(line)

    df['Distance_to_Nearest_Fault'] = df.apply(calculate_distance_to_nearest_fault, fault_lines=fault_lines, axis=1)

    missing_threshold = 0.3
    missing_values = df.isnull().mean()
    columns_to_drop = missing_values[missing_values > missing_threshold].index
    df.drop(columns_to_drop, axis=1, inplace=True)

    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', inplace=True)
    df['TimeToNext'] = (df['time'] - df['time'].shift(1)).dt.days
    df.loc[0, 'TimeToNext'] = df.loc[1, 'TimeToNext']
    df = df.iloc[1:]
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour

    df['rms'] = df['rms'].fillna(df['rms'].mean())
    df['magNst'] = df['magNst'].fillna(df['magNst'].median())

    df.to_csv(output_file, index=False)