import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the directory containing the CSV files
directory = 'RawData/IV_Spectra'

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
all_I_values = np.array(all_I_values)

# Calculate the average I values for each V
average_I_values = np.mean(all_I_values, axis=0)

# Use the first V_values as the common V axis (assuming all V_values are the same)
V_values = all_V_values[0]
# Fit V values and average I values to a cubic polynomial
coefficients, cov_matrix = np.polyfit(V_values, average_I_values, 3, cov=True)
cubic_fit = np.polyval(coefficients, V_values)

# Calculate the standard deviation errors on the coefficients
errors = np.sqrt(np.diag(cov_matrix))

# Print the coefficients and their errors
for i, (coef, err) in enumerate(zip(coefficients, errors)):
    print(f"Coefficient a{i}: {coef} ± {err}")
    # Print the cubic equation

equation = f"f(V) = {coefficients[0]:.3e}V^3 + {coefficients[1]:.3e}V^2 + {coefficients[2]:.3e}V + {coefficients[3]:.3e}"
print("Cubic equation of the fit:")
print(equation)

# Create a new figure
plt.figure(figsize=(10, 5))

# Plot average I vs V
plt.subplot(1, 2, 1)
plt.plot(V_values, average_I_values, 'bo', label=f'Averaged data from {file_count} spectra')  # 'bo' for blue points only
plt.plot(V_values, cubic_fit, 'r-', label='Cubic Fit of the Data')  # 'r-' for red line
plt.legend()
plt.grid(True)
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('Average I vs V')

# Calculate the derivative of the cubic fit
d_cubic_fit = np.polyder(coefficients)

# Evaluate the derivative at the V values
dIdV = np.polyval(d_cubic_fit, V_values)

# Calculate I/V
I_over_V = average_I_values / V_values

# Calculate (dI/dV)/(I/V)
dIdV_over_IoverV = dIdV / I_over_V
# Plot (dI/dV)/(I/V) vs V
plt.subplot(1, 2, 2)
plt.plot(V_values, dIdV_over_IoverV, 'ro', label='Derivative of fitted curve')  # 'ro' for red points only
plt.grid(True)
plt.legend()
plt.xlabel('Voltage (V)')
plt.ylabel('(dI/dV)/(I/V)')
plt.title('(dI/dV)/(I/V) vs V')

plt.tight_layout()
plt.savefig('Produced_Plots/IV_Spectra.png', dpi=300)
plt.show()
