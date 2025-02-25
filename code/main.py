import matplotlib.pyplot as plt
import numpy as np


def plot_raw(data):
    # Plot the data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b']
    labels = ['Target 1', 'Target 2', 'Target 3']

    for i in range(1, 4):
        subset = data[data[:, 0] == i]
        ax.scatter(subset[:, 1], subset[:, 2], subset[:, 3], c=colors[i-1], label=labels[i-1], alpha=0.7)

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("3D Plot of Target Positions")
    ax.legend()
    plt.show()
    
    
if __name__ == "__main__":
    # File name
    file_name = "data/tracker/anfahrt_pose1.txt"  # id, x, y, z
    # Read the file
    data = np.genfromtxt(file_name, delimiter=",", skip_header=1)
    
    # Plot the data
    plot_raw(data)