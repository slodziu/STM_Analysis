import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
#directory = 'RawData/IZ_Lines' #uncomment this line and comment the next line to run the code for IZ_Lines
directory = 'RawData/IZ_Gold'

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
all_Z_values = translated_Z_values_per_array


 # Plot the data

plt.figure(figsize=(10, 6))
for Z_values, I_values in zip(all_Z_values, all_I_values):
    plt.plot(Z_values, I_values, 'o', markersize=2, label='IZ Line')
plt.xlabel('Distance (m)')
plt.ylabel('Current (A)')
plt.title('I-Z Characteristics')
plt.legend()
plt.grid(True)
plt.savefig('Produced_Plots/IZ_Lines.png',dpi=300)
plt.show()

# Calculate the average I values for each Z value
average_I_values = [sum(i) / len(i) for i in zip(*all_I_values)]

# Calculate the average Z values (assuming all Z arrays are the same)
average_Z_values = [sum(z) / len(z) for z in zip(*all_Z_values)]

# Calculate the standard deviation of I values for each Z value
std_I_values = [np.std(i) for i in zip(*all_I_values)]

# Plot the average I-Z data with error bars



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
plt.savefig('Produced_Plots/Average_IZ_Line.png',dpi=300)
plt.show()
# Take the logarithm of the average Z and I values
log_average_Z_values = np.log10(np.abs(average_Z_values))
print(average_Z_values)
log_average_I_values = np.log10(average_I_values)

# Calculate the error in log scale
log_std_I_values = [np.log10(avg + std) - np.log10(avg) for avg, std in zip(average_I_values, std_I_values)]

# Plot the log-log average I-Z data with error bars
print(log_average_Z_values)
plt.plot(log_average_Z_values, log_average_I_values, 'r-', linewidth=2, label='Log Average IZ Line')
plt.fill_between(log_average_Z_values, 
                 [log_avg - log_std for log_avg, log_std in zip(log_average_I_values, log_std_I_values)], 
                 [log_avg + log_std for log_avg, log_std in zip(log_average_I_values, log_std_I_values)], 
                 color='purple', alpha=0.5, label='Log Std Dev Range')
plt.xlabel('Log Distance (log(m))')
plt.ylabel('Log Current (log(A))')
plt.title('Log-Log Average I-Z Characteristics')
plt.legend()
plt.grid(True)
plt.savefig('Produced_Plots/Log_Average_IZ_Line.png', dpi=300)
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
plt.savefig('Produced_Plots/Log_Average_IZ_Line_with_Fit.png', dpi=300)
plt.show()