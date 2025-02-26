import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_trajectory(data, plot_name):
    """
    Plot the 3D trajectory of the data.
    """
    # Extract coordinates
    X = data[:,1]
    Y = data[:,2]
    Z = data[:,3]

    # Create 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, color='b', s=5)  # Use scatter for dots

    # Labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title(plot_name)

    # Set equal scaling
    ax.set_box_aspect([1,1,1])  # Equal aspect ratio

    # Show plot
    plt.show()
    


if __name__ == '__main__':
    # File names
    file_names = [
        "data/tracker/Trajektoriezw1u2_vel50.txt",
        "data/tracker/Trajektoriezw1u2_vel100.txt",
    ]
    
    for file_name in file_names:
        # Load the data
        data_raw = pd.read_csv(file_name, sep=',', header=3)
        # Extract the relevant columns (2,3,4,5) [id, X, Y, Z]
        data = data_raw.iloc[:, [2, 3, 4, 5]].values
        data = np.array(data)
        
        # # Plot the data
        plot_name = file_name.split("/")[-1].split(".")[0]
        # plot_trajectory(data, plot_name)
        
        # Save data to .ply file
        # header
        header = "ply\nformat ascii 1.0\nelement vertex " + str(len(data)) + "\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        # data
        with open("data/results/" + plot_name + ".ply", "w") as f:
            f.write(header)
            for i in range(len(data)):
                f.write(str(data[i][1]) + " " + str(data[i][2]) + " " + str(data[i][3]) + "\n")
                
                
    