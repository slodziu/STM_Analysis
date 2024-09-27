import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the directory containing the CSV files
directory = 'RawData/IV_Spectra'

# Initialize lists to accumulate V and I values
all_V_values = []
all_I_values = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
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
all_I_values = np.array(all_I_values)

# Calculate the average I values for each V
average_I_values = np.mean(all_I_values, axis=0)

# Use the first V_values as the common V axis (assuming all V_values are the same)
V_values = all_V_values[0]

# Create a new figure
plt.figure(figsize=(10, 5))

# Plot average I vs V
plt.subplot(1, 2, 1)
plt.plot(V_values, average_I_values, 'bo')  # 'bo' for blue points only
plt.xlabel('Voltage (V)')
plt.ylabel('Current (I)')
plt.title('Average I vs V')

# Calculate dI/dV and (I/V)
dI_dV = np.gradient(average_I_values, V_values)
I_over_V = average_I_values / V_values
dIdV_over_IoverV = dI_dV / I_over_V

# Plot (dI/dV)/(I/V) vs V
plt.subplot(1, 2, 2)
plt.plot(V_values, dIdV_over_IoverV, 'ro')  # 'ro' for red points only
plt.xlabel('Voltage (V)')
plt.ylabel('(dI/dV)/(I/V)')
plt.title('(dI/dV)/(I/V) vs V')

plt.tight_layout()
plt.show()
