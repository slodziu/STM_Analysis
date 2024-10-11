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
Z_values = np.array(Z_values)*1e8
I_values = np.array(I_values)*1e9
log_Z_vals = [np.log(x) for x in Z_values]
log_I_values = [np.log(x) for x in I_values]

print(len(Z_values))
# Define the exponential model function
def exponential_model(Z, a, b, c):
    return a * np.exp(-b * Z) + c


# Fit the exponential model to the data with more lenient bounds
params, covariance = curve_fit(exponential_model, Z_values, I_values, maxfev=10000, p0=[0.26, 8, 0])

# Generate fitted I values using the model

print('Fit parameters', params)
# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(log_Z_vals, log_I_values, 'o', label='Data')
plt.xlabel('Tip Distance (m)')
plt.ylabel('Tunneling Current (A)')
plt.title('I vs Z Spectrum with Exponential Fit')
plt.legend()
plt.grid(True)
plt.show()