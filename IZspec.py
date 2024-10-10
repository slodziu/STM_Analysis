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
# Filter Z values greater than 5e-8
mask = Z_values > 5e-8
Z_values = Z_values[mask]
I_values = I_values[mask]
# Take the logarithm of both Z and I values
log_Z_values = np.log(Z_values)
log_I_values = np.log(I_values)
# Fit a linear model to the log-log data
linear_model = np.polyfit(log_Z_values, log_I_values, 1)
linear_fit = np.poly1d(linear_model)

# Generate fitted log(I) values using the linear model
log_I_fitted = linear_fit(log_Z_values)

# Calculate the errors of the fitted parameters
residuals = log_I_values - log_I_fitted
ss_res = np.sum(residuals**2)
ss_tot = np.sum((log_I_values - np.mean(log_I_values))**2)
r_squared = 1 - (ss_res / ss_tot)
p_cov = np.polyfit(log_Z_values, log_I_values, 1, cov=True)[1]
errors = np.sqrt(np.diag(p_cov))

print('R-squared:', r_squared)
print('Slope:', linear_model[0], '±', errors[0])
print('Intercept:', linear_model[1], '±', errors[1])
# Plot the linear fit on top of the log-log data
plt.plot(log_Z_values, log_I_fitted, '-', label='Linear fit: log(I) = {:.2f} * log(Z) + {:.2f}'.format(linear_model[0], linear_model[1]))

# Plot the log-log data
plt.plot(log_Z_values, log_I_values, 'o', label='log(I) vs log(Z)')
plt.xlabel('log(Tip Distance) (log(m))')
plt.ylabel('log(Tunneling Current) (log(A))')
plt.title('log(I) vs log(Z) Spectrum')
plt.legend()
plt.grid(True)
plt.savefig('Produced_Plots\IZspec.png', dpi=300)
plt.show()
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
plt.xlabel('Tip Distance (m)')
plt.ylabel('Tunneling Current (A)')
plt.title('I vs Z Spectrum with Exponential Fit')
plt.legend()
plt.grid(True)
plt.show()