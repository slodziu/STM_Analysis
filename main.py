import numpy as np
class Line:
    def __init__(self, x0, y0, x1, y1):
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
        return self.points
# Function to read data from a text file
def read_data_from_file(filename):
    # Read the file and skip the header line
    data = np.loadtxt(filename, skiprows=1)
    return data


# Function to calculate midpoint of a line
def calculate_midpoint(x0, y0, x1, y1):
    return [(x0 + x1) / 2, (y0 + y1) / 2]

# Function to calculate Euclidean distance between two points
def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# Main function to compute average distance and error
def calculate_average_distance_with_error(filename):
    N = 1000  # Number of points to generate on each line
    # Read data from file
    data = read_data_from_file(filename)
    print('Here')
    print(data[0][1])
    # List to store distances between consecutive lines
    lines = []
    
    # Generate points on each line
    for j in range(len(data)):
        newLine = Line(data[j][0],data[j][1],data[j][2],data[j][3])
        newLine.set_slope()
        newLine.set_points(N)
        lines.append(newLine)
    interline_distances = []
    interline_distances_err = []
    for i in range(len(lines)-1):
        distances = []
        curr_slope = lines[i].get_slope()
        perp_slope = -1/curr_slope
        adjacent_line_slope = lines[i+1].get_slope()
        adjacent_line_y_intercept = lines[i+1].y0 - adjacent_line_slope*lines[i+1].x0
        for point in lines[i].get_points():
            perp_y_intercept = point[1] - perp_slope*point[0]
            x_meeting = (perp_y_intercept - adjacent_line_y_intercept)/(adjacent_line_slope - perp_slope)
            y_meeting = perp_slope*x_meeting + perp_y_intercept
            new_distance = calculate_distance(point,[x_meeting,y_meeting])
            distances.append(new_distance)
        print('Average distance between line',i+1,'and line',i+2,':',np.mean(distances))
        interline_distances.append(np.mean(distances))
        interline_distances_err.append(np.std(distances)/np.sqrt(N))
    print(interline_distances)
    print(interline_distances_err)
# Example usage
filename = 'RawData\slopes3.txt'  # Replace this with the path to your data file
calculate_average_distance_with_error(filename)