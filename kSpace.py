import numpy as np
import os

# Define the directory containing the files
directory_path = 'RawData/K_Vals'

# Initialize an empty list to store the 4th column values
fourth_column_values = []

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    # Open each file and read the lines
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into columns
            columns = line.split()
            # Check if the line has at least 4 columns
            if len(columns) >= 4:
                
                # Append the 4th column value to the list
                fourth_column_values.append(columns[3])


print(fourth_column_values)
# Convert the list of strings to a list of floats
fourth_column_values = [float(value) for value in fourth_column_values]

# Calculate the mean
mean_value = np.mean(fourth_column_values)

# Calculate the standard error of the mean
standard_error = np.std(fourth_column_values) / np.sqrt(len(fourth_column_values))
# Print the results in scientific notation

a = 0.2461e-9 #lattice parameter in meters
k = 4 * np.pi / (a*np.sqrt(3))
#k = 2*np.pi/a 
print(f'Desired value: {k:.2e}')
print(f"Mean: {mean_value:.2e}")
print(f"Standard Error: {standard_error:.2e}")
diff = abs(mean_value - k/2)
sigma = diff/standard_error
print(f"This many sigma away: {sigma}")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the image
image_path = os.path.join('RawData', 'k-vec.png')
image = mpimg.imread(image_path)

# Create a figure and axis
fig, ax = plt.subplots()

# Display the image
ax.imshow(image)

# Set the x and y axis labels
ax.set_xlabel('X-axis label')
ax.set_ylabel('Y-axis label')

# Save the figure with axes to the Produced_Plots directory
output_path = os.path.join('Produced_Plots', 'k-vec_with_axes.png')
os.makedirs('Produced_Plots', exist_ok=True)
plt.savefig(output_path)

# Show the plot (optional)
plt.show()