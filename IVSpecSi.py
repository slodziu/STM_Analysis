import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the directory containing the CSV files
directory = 'RawData/IV_Si'

# Initialize lists to accumulate V and I values
all_V_values = []
all_I_values = []
file_count = 0
# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_count += 1
        filepath = os.path.join(directory, filename)
        # Read the CSV file without headers
        data = pd.read_csv(filepath, header=None, delimiter=';')
        
        # Extract data
        V_values = data.iloc[0].tolist()
        I_values = data.iloc[1].tolist()
        
        # Append to the lists
        all_V_values.append(V_values)
        all_I_values.append(I_values)
        # Convert lists to numpy arrays for easier manipulation
all_V_values = np.array(all_V_values)
all_I_values = np.array(all_I_values)*1e10
# Shift I values such that data goes through (0,0)
for i in range(len(all_I_values)):
    all_I_values[i] += all_I_values[i][0]
        # Calculate the average and standard deviation
average_V = np.mean(all_V_values, axis=0)
average_I = np.mean(all_I_values, axis=0)
std_I = np.std(all_I_values, axis=0)

# Plot the average I-V curve with error bars
plt.figure(figsize=(10, 6))
plt.plot(average_V, average_I, label='Average I-V Curve')
plt.fill_between(average_V, average_I - std_I, average_I + std_I, color='b', alpha=0.2, label='Standard Deviation')
plt.xlabel('Tip Voltage (V)')
plt.ylabel('Tip Current (A)')
plt.title('Average I-V Curve with Error Bars')
plt.legend()
plt.grid(True)
plt.savefig('Produced_Plots/Silicon/IV_Si_Average.png',dpi=300)
plt.show()

# Define the fitting function
def fit_function(V, I_0, A, C):
    return I_0 * (np.exp(A * V) - 1) + C

# Perform the curve fitting
popt, pcov = curve_fit(fit_function, average_V, average_I, p0=[1e-10, 1, 0])

# Extract the fitting parameters and their errors
I_0_fit, A_fit, C_fit = popt
errors = np.sqrt(np.diag(pcov))

# Calculate the fitted curve
fit = fit_function(average_V, I_0_fit, A_fit, C_fit)


plt.plot(average_V, average_I, label=f'Averaged data from {file_count} spectra')  # 'bo' for blue points only
plt.plot(average_V, fit, 'r-', label='Cubic Fit of the Data')  # 'r-' for red line
plt.fill_between(average_V,fit - errors[0],fit + errors[0], color='r', alpha=0.2, label='Standard Deviation')
plt.legend()
plt.grid(True)
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('Average I vs V')
plt.show()