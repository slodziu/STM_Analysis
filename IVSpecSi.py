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

# Plot the average I-V curve with errors
plt.figure(figsize=(10, 6))
plt.plot(average_V, average_I, label='Average I-V Curve',color='r')
plt.fill_between(average_V, average_I - std_I, average_I + std_I, color='b', alpha=0.5, label='Standard Deviation')
plt.xlabel('Tip Voltage (V)')
plt.ylabel('Tip Current (A)')
plt.title('Average I-V Curve with Error Bars')
plt.legend()
plt.grid(True)
plt.savefig('Produced_Plots/Silicon/IV_Si_Average.png',dpi=300)
plt.show()
# Filter the average I and V to where average V > 0
positive_indices = average_V > 0
average_V = average_V[positive_indices]
average_I = average_I[positive_indices]
std_I = std_I[positive_indices]
# Define a cubic function
def cubic_function(V, a, b, c, d):
    return a*V**3 + b*V**2 + c*V + d

# Fit the average I-V data to the cubic function
params, covariance = curve_fit(cubic_function, average_V, average_I)

# Calculate the standard deviation errors on the parameters
errors = np.sqrt(np.diag(covariance))

# Print the fit equation with parameters
print(f"Fitted cubic equation: I = ({params[0]:.3e})*V^3 + ({params[1]:.3e})*V^2 + ({params[2]:.3e})*V + ({params[3]:.3e})")
print(f"Parameter errors: {errors}")

# Generate the fitted curve
fit = cubic_function(average_V, *params)
plt.scatter(average_V, average_I, label=f'Averaged data from {file_count} spectra', s=2)  # 's=10' sets the size of the points
plt.plot(average_V, fit, 'r-', label='Cubic Fit of the Data')  # 'r-' for red line
plt.fill_between(average_V,fit - errors[0],fit + errors[0], color='r', alpha=0.3, label='Standard Deviation')
plt.legend()
plt.grid(True)
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('Average I vs V')
plt.savefig('Produced_Plots/Silicon/IV_Si_Fit.png', dpi=300)
plt.show()

# Calculate the derivative of the cubic fit
d_cubic_fit = np.polyder(params)

# Evaluate the derivative at the V values
dIdV = np.polyval(d_cubic_fit, average_V)

# Calculate I/V
I_over_V = average_I /average_V

# Calculate (dI/dV)/(I/V)
dIdV_over_IoverV = dIdV / I_over_V

# Calculate the error on (dI/dV)/(I/V)
dIdV_error = np.polyval(np.polyder(np.polyfit(average_V, std_I, 3)), average_V)
I_over_V_error = std_I / average_V
dIdV_over_IoverV_error = np.sqrt((dIdV_error / I_over_V)**2 + (dIdV * I_over_V_error / I_over_V**2)**2)

# Plot (dI/dV)/(I/V) with error bars
plt.plot(average_V, dIdV_over_IoverV, label='Derivative of fitted curve')  # 'ro' for red points only
plt.fill_between(average_V, dIdV_over_IoverV - dIdV_over_IoverV_error, dIdV_over_IoverV + dIdV_over_IoverV_error, color='r', alpha=0.5, label='Error')
plt.grid(True)
plt.legend()
plt.xlabel('Voltage (V)')
plt.ylabel('(dI/dV)/(I/V)')
plt.title('(dI/dV)/(I/V) vs V')
plt.savefig('Produced_Plots/Silicon/IV_Si_dIdV_over_IoverV.png', dpi=300)
plt.show()
