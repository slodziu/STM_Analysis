import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the directory containing the CSV files
directory = 'RawData/IV_Gold'

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
# Fit a linear model to the average I-V data
coefficients = np.polyfit(average_V, average_I, 1)
m, c = coefficients
print(m,c)
# Generate fitted I values using the linear model
fitted_I = m * average_V + c

# Plot the fitted line

# Plot the average I-V curve with error bars
plt.figure(figsize=(10, 6))
plt.plot(average_V, average_I, label='Average I-V Curve')
plt.plot(average_V, fitted_I, label=f'Linear Fit', color='r')
plt.fill_between(average_V, average_I - std_I, average_I + std_I, color='b', alpha=0.2, label='Standard Deviation')
plt.xlabel('Tip Voltage (V)')
plt.ylabel('Tip Current (A)')
plt.title('Average I-V Curve with Error Bars')
plt.legend()
plt.grid(True)
plt.savefig('Produced_Plots/Gold/IV_Gold_LinearFit.png',dpi=300)
plt.show()

# Calculate the derivative of the linear fit
derivative = np.gradient(fitted_I, average_V)
I_over_V = fitted_I / V_values
dIdV_over_IoverV = derivative/ I_over_V
# Calculate the error on the derivative of the linear fit
std_derivative = np.gradient(std_I, average_V)
std_I_over_V = std_I / average_V
std_dIdV_over_IoverV = np.sqrt((std_derivative / I_over_V)**2 + (derivative * std_I_over_V / I_over_V**2)**2)

# Plot the derivative of the linear fit with error bars
plt.figure(figsize=(10, 6))
plt.plot(average_V, dIdV_over_IoverV, label='Derivative of Linear Fit', color='g')
plt.fill_between(average_V, dIdV_over_IoverV - std_dIdV_over_IoverV, dIdV_over_IoverV + std_dIdV_over_IoverV, color='g', alpha=0.2, label='Standard Deviation')
plt.xlabel('Voltage (V)')
plt.ylabel('dln(I)/dln(V)')
plt.title('Derivative of Linear Fit/(I-V ratio) with Error Bars')
plt.legend()
plt.grid(True)
plt.savefig('Produced_Plots/Gold/IV_Gold.png',dpi=300)
plt.show()
# Plot the derivative of the linear fit with error bars for 0.2 < V < 0.4
plt.figure(figsize=(10, 6))
plt.plot(average_V, dIdV_over_IoverV, label='Derivative of Linear Fit', color='g')
plt.fill_between(average_V, dIdV_over_IoverV - std_dIdV_over_IoverV, dIdV_over_IoverV + std_dIdV_over_IoverV, color='g', alpha=0.2, label='Standard Deviation')
plt.xlabel('Voltage (V)')
plt.ylabel('dln(I)/dln(V)')
plt.title('Derivative of Linear Fit with Error Bars (0.2 < V < 0.4)')
plt.legend()
plt.grid(True)
plt.xlim(0.2, 0.4)
plt.ylim(-200, 200)
plt.savefig('Produced_Plots/Gold/IV_Gold_0.2-0.4.png',dpi=300)
plt.show()

# Define the sinx + x function
def sinx_x(V, a, b, c, d):
    return a * np.cos(b * V) + c*V +d

# Fit the average I-V data to the sinx * x function
params, params_covariance = curve_fit(sinx_x, average_V, average_I)

# Extract the parameters
a, b, c, d = params
print(a, b, c, d)

# Generate fitted I values using the sinx * x model
fitted_I_sinx_x = sinx_x(average_V, a, b, c, d)

# Plot the fitted sinx * x function
plt.figure(figsize=(10, 6))
plt.plot(average_V, average_I, label='Average I-V Curve')
plt.plot(average_V, fitted_I_sinx_x, label=f'Sinx + x Fit', color='r')
plt.fill_between(average_V, average_I - std_I, average_I + std_I, color='b', alpha=0.2, label='Standard Deviation')
plt.xlabel('Tip Voltage (V)')
plt.ylabel('Tip Current (A)')
plt.title('Average I-V Curve with Sinx + x Fit')
plt.legend()
plt.grid(True)
plt.savefig('Produced_Plots/Gold/IV_Gold_Sinx_x_Fit.png', dpi=300)
plt.show()