import numpy as np
import os
class Line:
    def __init__(self, x0, y0, x1, y1):
        """
        Initializes the coordinates of a rectangle.

        Parameters:
        x0 (float): The x-coordinate of the first point.
        y0 (float): The y-coordinate of the first point.
        x1 (float): The x-coordinate of the second point.
        y1 (float): The y-coordinate of the second point.
        """
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
    def set_slope(self):
        """
        Set the slope of the line.
        """
        if (self.x1 - self.x0) == 0:
            self.slope= np.inf  # Handle vertical line case
        self.slope = (self.y1 - self.y0) / (self.x1 - self.x0)
    def get_slope(self):
        """
        Retrieve the slope value.
        Returns:
            float: The slope value.
        """

        return self.slope
    def set_points(self,N=1000):
        """
        Generate N points along the line between (x0, y0) and (x1, y1).

        Returns:
        np.array -- An array of shape (N, 2) with N points along the line
        """
        x_points = np.linspace(self.x0, self.x1, N)
        y_points = np.linspace(self.y0, self.y1, N)
        points = np.column_stack((x_points, y_points))
        self.points = points
    def get_points(self):
        """
        Retrieve the list of points.

        Returns:
            list: A list containing the points.
        """
        return self.points
# Function to read data from a text file
def read_data_from_file(filename):
    """
    Reads numerical data from a text file, skipping the first row.
    Parameters:
    filename (str): The path to the file containing the data.
    Returns:
    numpy.ndarray: A NumPy array containing the data read from the file.
    """
    
    data = np.loadtxt(filename, skiprows=1)
    return data

def calculate_midpoint(x0, y0, x1, y1):
    """
    Calculate the midpoint between two points (x0, y0) and (x1, y1).

    Args:
        x0 (float): The x-coordinate of the first point.
        y0 (float): The y-coordinate of the first point.
        x1 (float): The x-coordinate of the second point.
        y1 (float): The y-coordinate of the second point.

    Returns:
        list: A list containing the x and y coordinates of the midpoint.
    """
    return [(x0 + x1) / 2, (y0 + y1) / 2]


def calculate_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    p1 (tuple): The first point as a tuple of (x, y) coordinates.
    p2 (tuple): The second point as a tuple of (x, y) coordinates.

    Returns:
    float: The Euclidean distance between the two points.
    """
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def calculate_average_distance_with_error(filename):
    """
    Calculate the average distance and its error between consecutive lines from data in a file.
    This function reads line data from a specified file, generates points on each line, and calculates
    the average distance between consecutive lines along with the associated error.
    Args:
        filename (str): The path to the file containing line data. The file should contain data in a format
                        that can be read by the `read_data_from_file` function.
    Returns:
        None: This function prints the average interline distance and the error in the average interline distance.
    Notes:
        - The function assumes that the `read_data_from_file`, `Line`, and `calculate_distance` functions/classes
          are defined elsewhere in the code.
        - The number of points generated on each line is set to 1000.
        - The average distance and error are printed in nanometers (nm).
    """
    N = 1000  # Number of points to generate on each line
    # Read data from file
    data = read_data_from_file(filename)
    print('Here')
    print(data[0][1])
    # List to store distances between consecutive lines
    lines = []
    
    # Generate points on each line
    for j in range(len(data)):
        newLine = Line(data[j][0], data[j][1], data[j][2], data[j][3])
        newLine.set_slope()
        newLine.set_points(N)
        lines.append(newLine)

    interline_distances = []
    interline_distances_err = []

    for i in range(len(lines) - 1): 
        distances = []
        curr_slope = lines[i].get_slope()
        perp_slope = -1 / curr_slope # Slope of the perpendicular line
        adjacent_line_slope = lines[i + 1].get_slope()  # Slope of the adjacent line
        adjacent_line_y_intercept = lines[i + 1].y0 - adjacent_line_slope * lines[i + 1].x0 # y-intercept of the adjacent line

        for point in lines[i].get_points(): 
            perp_y_intercept = point[1] - perp_slope * point[0]
            x_meeting = (perp_y_intercept - adjacent_line_y_intercept) / (adjacent_line_slope - perp_slope) 
            y_meeting = perp_slope * x_meeting + perp_y_intercept
            new_distance = calculate_distance(point, [x_meeting, y_meeting]) 
            distances.append(new_distance)

        print(f'Average distance between line {i + 1} and line {i + 2}: {np.mean(distances)}')
        interline_distances.append(np.mean(distances))
        interline_distances_err.append(np.std(distances) / np.sqrt(N))

    average_interline_distance = np.mean(interline_distances)
    average_interline_distance_err = np.sqrt(np.sum(np.square(interline_distances_err)))

    print(f'Average interline distance: {average_interline_distance / 1e-9} nm')
    print(f'Error in average interline distance: {average_interline_distance_err / 1e-9} nm')
def process_lines_in_folder(folder_path):
        """
        Process all line data files in a specified folder and create Line objects.

        Args:
            folder_path (str): The path to the folder containing line data files.

        Returns:
            list: A list of Line objects created from the files in the folder.
        """
        line_objects = []

        for filename in os.listdir(folder_path):
            line_pair = []
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                data = read_data_from_file(file_path)
                for j in range(len(data)):
                    new_line = Line(data[j][0], data[j][1], data[j][2], data[j][3])
                    new_line.set_slope()
                    new_line.set_points()
                    line_pair.append(new_line) 
                line_objects.append(line_pair)

        return line_objects

    # Example usage
folder_path = 'RawData/PerpLines'
lines = process_lines_in_folder(folder_path)
for line_pair in lines:
    # Calculate the angle between the two lines
    slope1 = line_pair[0].get_slope()
    slope2 = line_pair[1].get_slope()
    angle = np.arctan(abs((slope2 - slope1) / (1 + slope1 * slope2))) * (180 / np.pi)
    print(f'Angle between lines: {angle} degrees')

    # Calculate the lengths of the lines
    length1 = calculate_distance((line_pair[0].x0, line_pair[0].y0), (line_pair[0].x1, line_pair[0].y1))
    length2 = calculate_distance((line_pair[1].x0, line_pair[1].y0), (line_pair[1].x1, line_pair[1].y1))
    length_ratio = max(length1, length2) / min(length1, length2)
    print(f'Ratio of lengths (longer/shorter): {length_ratio}')
    print('Compared to square root of 3:', np.sqrt(3))
    # Calculate the average of the ratio of the lengths with the error
    length_ratios = []
    angles = []

for line_pair in lines:
        length1 = calculate_distance((line_pair[0].x0, line_pair[0].y0), (line_pair[0].x1, line_pair[0].y1))
        length2 = calculate_distance((line_pair[1].x0, line_pair[1].y0), (line_pair[1].x1, line_pair[1].y1))
        length_ratio = max(length1, length2) / min(length1, length2)
        length_ratios.append(length_ratio)

        slope1 = line_pair[0].get_slope()
        slope2 = line_pair[1].get_slope()
        angle = np.arctan(abs((slope2 - slope1) / (1 + slope1 * slope2))) * (180 / np.pi)
        angles.append(angle)

average_length_ratio = np.mean(length_ratios)
length_ratio_error = np.std(length_ratios) / np.sqrt(len(length_ratios))
average_angle = np.mean(angles)
angle_error = np.std(angles) / np.sqrt(len(angles))

print(f'Average ratio of lengths: {average_length_ratio}')
print(f'Error in average ratio of lengths: {length_ratio_error}')
print(f'Average angle: {average_angle} degrees')
print(f'Error in average angle: {angle_error} degrees')
filename = 'RawData\slopes3.txt'
second_filename = 'RawData\slopes4.txt'
#calculate_average_distance_with_error(filename)
calculate_average_distance_with_error(second_filename) 