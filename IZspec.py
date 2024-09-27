import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the directory containing the CSV files
directory = 'RawData'

# Initialize lists to store data
I_values = []
Z_values = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)
        # Read the CSV file without headers
        data = pd.read_csv(filepath, header=None, delimiter=';')
        # Append data to lists
        Z_values.extend(data.iloc[0].tolist())
        I_values.extend(data.iloc[1].tolist())

# Convert lists to numpy arrays for fitting
Z_values = np.array(Z_values)
I_values = np.array(I_values)
print(len(Z_values))
# Define the exponential model function
def exponential_model(Z, a, b, c):
    return a * np.exp(b * Z) + c

# Fit the exponential model to the data with more lenient bounds
params, covariance = curve_fit(exponential_model, Z_values, I_values, bounds=(-np.inf, np.inf))

# Generate fitted I values using the model
I_fitted = exponential_model(Z_values, *params)
print('Fit parameters', params)
# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(Z_values, I_values, 'o', label='I vs Z')
plt.plot(Z_values, I_fitted, '-', label='Fitted model')
plt.xlabel('Z')
plt.ylabel('I')
plt.title('I vs Z Spectrum with Exponential Fit')
plt.legend()
plt.grid(True)
plt.show()