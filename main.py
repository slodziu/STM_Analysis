import numpy as np
import pandas as pd

def calculate_average_distance(file_path):
    # Specify the correct column names
    column_names = ['Δx [pm]', 'Δy [pm]', 'φ [deg]', 'R [pm]', 'Δz [pA]']
    
    # Read the data, skipping over any unwanted columns
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=column_names, usecols=[0, 2, 4, 6, 8])

    # Convert Δx and Δy to numeric values, forcing errors to NaN
    data['Δx [pm]'] = pd.to_numeric(data['Δx [pm]'], errors='coerce')
    data['Δy [pm]'] = pd.to_numeric(data['Δy [pm]'], errors='coerce')

    # Drop rows where Δx or Δy contains NaN values (i.e., non-numeric data)
    data = data.dropna(subset=['Δx [pm]', 'Δy [pm]'])

    # Extract Δx and Δy columns as numeric arrays
    delta_x = data['Δx [pm]'].values
    delta_y = data['Δy [pm]'].values

    # Check if there are at least two points to calculate differences
    if len(delta_x) < 2 or len(delta_y) < 2:
        raise ValueError("Not enough valid data points to calculate distances")

    # Calculate distances between consecutive lines using Euclidean distance
    distances = np.sqrt(np.diff(delta_x)**2 + np.diff(delta_y)**2)

    # Calculate the average distance and the error (standard error of the mean)
    avg_distance = np.mean(distances)
    std_error = np.std(distances, ddof=1) / np.sqrt(len(distances))

    return avg_distance, std_error

# Provide the full path to slopes.txt
file_path = 'slopes.txt'  # Replace with the correct file path
avg_distance, std_error = calculate_average_distance(file_path)
print(f"Average Distance: {avg_distance/1000:.4f} nm")
print(f"Error (Standard Error): {std_error/1000:.4f} nm")
