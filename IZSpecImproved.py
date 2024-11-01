import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress

import matplotlib.pyplot as plt
directory = 'RawData/IZ_Si'

all_Z_values = []
all_I_values = []
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)
        # Read the CSV file without headers
        data = pd.read_csv(filepath, header=None, delimiter=';')
        
        # Extract data
        V_values = data.iloc[0].tolist()
        I_values = data.iloc[1].tolist()
        all_Z_values.append(V_values)
        all_I_values.append(I_values)
        
        # Flatten the lists of lists
        # Find the smallest Z value across all lists
min_Z_value = min(min(sublist) for sublist in all_Z_values)
# Calculate the minimum Z value for each Z array and store it
min_Z_values_per_array = [min(sublist) for sublist in all_Z_values]
# Translate Z values so that they all start at the same value
translated_Z_values = [[z - min_Z_value for z in sublist] for sublist in all_Z_values]
all_Z_values = translated_Z_values
# Translate each Z array by its own minimum Z value
translated_Z_values_per_array = [
    [z - min_z for z in sublist] 
    for sublist, min_z in zip(all_Z_values, min_Z_values_per_array)
]
# Translate each I array so that they all start at 0
translated_I_values = [
    [i - sublist[0] for i in sublist] 
    for sublist in all_I_values
]
all_I_values = translated_I_values
all_Z_values = translated_Z_values_per_array


 # Plot the data

plt.figure(figsize=(10, 6))
plot_count = 1
for Z_values, I_values in zip(all_Z_values, all_I_values):
    plt.plot(Z_values, I_values, 'o', markersize=2, label=f'n={plot_count}')
    plot_count += 1
plt.xlabel('Distance (m)')
plt.ylabel('Current (A)')
plt.title(f'Raw I-Z Data from {plot_count} spectra')
plt.legend()
plt.grid(True)
plt.savefig('Produced_Plots/Gold/IZ_Lines.png',dpi=300)
plt.show()

# Filter out the values for Z > -8.645e-06
filtered_Z_values = []
filtered_I_values = []
for Z_values, I_values in zip(all_Z_values, all_I_values):
    filtered_Z = []
    filtered_I = []
    for z, i in zip(Z_values, I_values):
        if z >= -8.645e-08:
            filtered_Z.append(z)
            filtered_I.append(i)
    filtered_Z_values.append(filtered_Z)
    filtered_I_values.append(filtered_I)

#all_Z_values = filtered_Z_values
#all_I_values = filtered_I_values

# Calculate the average I values for each Z value
average_I_values = [sum(i) * 1e9/ len(i) for i in zip(*all_I_values)] 

# Calculate the average Z values (assuming all Z arrays are the same)
average_Z_values = [sum(z) * 1e9 / len(z) for z in zip(*all_Z_values)] 

# Calculate the standard deviation of I values for each Z value
std_I_values = [np.std(i)* 1e9 for i in zip(*all_I_values)]

# Plot the average I-Z data with error bars

# Shift all Z values so that they start at 0
shifted_Z_values = [[z - min(average_Z_values) for z in average_Z_values]]
shifted_I_values = [[i - min(average_I_values) for i in average_I_values]]

# Update average_Z_values and average_I_values
average_Z_values = shifted_Z_values[0]
average_I_values = shifted_I_values[0]

# Plot the average I-Z data
plt.plot(average_Z_values, average_I_values, 'r-', linewidth=2, label='Average IZ Line')
plt.fill_between(average_Z_values, 
                 [avg - std for avg, std in zip(average_I_values, std_I_values)], 
                 [avg + std for avg, std in zip(average_I_values, std_I_values)], 
                 color='purple', alpha=0.5, label='Std Dev Range')
plt.xlabel('Distance (m)')
plt.ylabel('Current (A)')
plt.title('Average I-Z Characteristics')
plt.legend()
plt.grid(True)
plt.savefig('Produced_Plots/Silicon/Average_IZ_Line.png',dpi=300)
plt.show()
# Take the logarithm of the average Z and I values
log_average_Z_values = np.log10(average_Z_values)
print(average_Z_values)
log_average_I_values = np.log10(np.abs(average_I_values))
# Filter log Z values where log(Z) > 0 and shorten the I array accordingly
filtered_log_Z_values = []
filtered_log_I_values = []
for log_Z, log_I in zip(log_average_Z_values, log_average_I_values):
    if log_Z > 0:
        filtered_log_Z_values.append(log_Z)
        filtered_log_I_values.append(log_I)

log_average_Z_values = filtered_log_Z_values
log_average_I_values = filtered_log_I_values
# Calculate the error in log scale
log_std_I_values = [np.abs(np.log10(avg + std) - np.log10(avg)) for avg, std in zip(average_I_values, std_I_values)]
# Ensure log_average_Z_values and log_std_I_values are of the same length
min_length = min(len(log_average_Z_values), len(log_std_I_values))
log_average_Z_values = log_average_Z_values[:min_length]
log_std_I_values = log_std_I_values[:min_length]

# Interpolate the log_std_I_values to make it smooth
interp_func = interp1d(log_average_Z_values, log_std_I_values, kind='linear', fill_value='extrapolate')

# Generate smooth log_std_I_values
log_std_I_values = interp_func(log_average_Z_values)

# Plot the log-log average I-Z data with error bars
plt.plot(log_average_Z_values, log_average_I_values, 'r-', linewidth=2, label='Log Average IZ Line')
plt.fill_between(log_average_Z_values, 
                 [log_avg - log_std for log_avg, log_std in zip(log_average_I_values, log_std_I_values)], 
                 [log_avg + log_std for log_avg, log_std in zip(log_average_I_values, log_std_I_values)], 
                 color='purple', alpha=0.5, label='Log Std Dev Range')
plt.xlabel(r'Log Distance ($\log(m)$)')
plt.ylabel(r'Log Current ($\log(A)$)')
plt.title('Log-Log Average I-Z Characteristics')
plt.legend()
plt.grid(True)
plt.savefig('Produced_Plots/Silicon/Log_Average_IZ_Line.png', dpi=300)
plt.show()
# Perform a linear fit to the log-log data
slope, intercept = np.polyfit(log_average_Z_values, log_average_I_values, 1)

# Generate fitted line data
fitted_log_I_values = [slope * z + intercept for z in log_average_Z_values]

# Plot the log-log average I-Z data with the fitted line
plt.plot(log_average_Z_values, log_average_I_values, 'o', markersize=2, label='Log Average IZ Line')
plt.plot(log_average_Z_values, fitted_log_I_values, 'r-', linewidth=2, alpha=0.7, label=f'Fitted Line: y={slope:.2f}x+{intercept:.2f}')
plt.fill_between(log_average_Z_values, 
                 [log_avg - log_std for log_avg, log_std in zip(log_average_I_values, log_std_I_values)], 
                 [log_avg + log_std for log_avg, log_std in zip(log_average_I_values, log_std_I_values)], 
                 color='grey', alpha=0.5, label='Log Std Dev Range')
plt.xlabel('Log Distance (log(m))')
plt.ylabel('Log Current (log(A))')
plt.title('Log-Log Average I-Z Characteristics with Linear Fit')
plt.legend()
plt.grid(True)
plt.savefig('Produced_Plots/Silicon/Log_Average_IZ_Line_with_Fit.png', dpi=300)
plt.show() 

log_average_Z_values = np.array(log_average_Z_values)
log_average_I_values = np.array(log_average_I_values)


valid_indices = np.where(log_average_Z_values >= 0.8)
possible_Z_vals = []
valid_Z = log_average_Z_values[valid_indices]
valid_I = log_average_I_values[valid_indices]
# Perform a linear fit to the valid log-log data
valid_slope, valid_intercept = np.polyfit(valid_Z, valid_I, 1)

# Generate fitted line data for the valid range
valid_fitted_I = valid_slope * valid_Z + valid_intercept

plt.plot(valid_Z,valid_I,'o',markersize = 2,label='Log Average IZ Line')
plt.plot(valid_Z,valid_fitted_I,label='Fitted Line')
plt.fill_between(valid_Z,valid_I-log_std_I_values[valid_indices],valid_I+log_std_I_values[valid_indices],color='purple',alpha=0.5,label='Log Std Dev Range')
plt.xlabel('Log Distance (log(m))')
plt.ylabel('Log Current (log(A))')
plt.title('Log-Log Average I-Z Characteristics with Linear Fit')
plt.legend()
plt.grid(True)
plt.savefig('Produced_Plots/Silicon/Log_Average_IZ_Line_with_Fit_Valid.png', dpi=300)
plt.show()

# Calculate the errors in the fit parameters

slope, intercept, r_value, p_value, std_err = linregress(valid_Z, valid_I)

# Print the fit line parameters with their errors
print(f"Slope: {slope:.4f} ± {std_err:.4f}")
print(f"Intercept: {intercept:.4f}")