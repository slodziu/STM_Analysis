import os
import pandas as pd

import matplotlib.pyplot as plt

directory = 'RawData/IZ_Lines'
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

        
