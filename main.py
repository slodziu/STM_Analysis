import numpy as np
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

filename = 'RawData\slopes3.txt'
second_filename = 'RawData\slopes4.txt'
#calculate_average_distance_with_error(filename)
calculate_average_distance_with_error(second_filename)