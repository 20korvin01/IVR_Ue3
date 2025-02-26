import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    
def pos_repeat(data):
    """
    Positioning repeatability of the 3D points. ~ ISO 9283: 1998 (E)
    """
    # Extract x, y, z
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]
    # Compute the mean of the x, y, z coordinates
    mean_x = np.mean(xs) # x_bar
    mean_y = np.mean(ys) # y_bar
    mean_z = np.mean(zs) # z_bar
    # Compute distance of each point to the m ean
    distances = np.sqrt((xs - mean_x)**2 + (ys - mean_y)**2 + (zs - mean_z)**2)  # l_j
    # Compute the mean of the distances
    mean_distance = np.mean(distances)  # l_bar
    # Compute S_l
    std_l = np.sqrt(np.sum((distances - mean_distance)**2) / (data.shape[0] - 1))  # S_l
    # Compute the positioning repeatability
    pos_repeat = mean_distance + 3 * std_l  # RP_l
    
    return pos_repeat
    


if __name__ == '__main__':
    # File names
    file_names = [
        "data/tracker/Pose1Stationaer_fest.txt",
        "data/tracker/Pose1Stationaer_freedrive.txt",
        "data/tracker/PoseAussenStationaer_fest.txt",
        "data/tracker/PoseAussenStationaer_freedrive.txt",
    ]
    
    # store positioning repeatability
    pos_repeats = []
    
    for file_name in file_names:
        # Load the data
        data_raw = pd.read_csv(file_name, sep=',', header=3)
        # Extract the relevant columns (3,4,5) [X, Y, Z]
        data = data_raw.iloc[:, [3, 4, 5]].values
        data = np.array(data)
        
        # Compute the positioning repeatability
        pos_rep = pos_repeat(data)
        pos_repeats.append(pos_rep)
        print("Positioning repeatability of " + file_name + " is: " + str(pos_rep))
        
        
        
        
        # Save data to .ply file
        # header
        header = "ply\nformat ascii 1.0\nelement vertex " + str(len(data)) + "\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        # data
        with open("data/results/" + file_name.split("/")[-1].split(".")[0] + ".ply", "w") as f:
            f.write(header)
            for i in range(len(data)):
                f.write(str(data[i][0]) + " " + str(data[i][1]) + " " + str(data[i][2]) + "\n")
                
                
    # Save positioning repeatability to a file
    with open("data/results/kin_stat_pos_repeat.txt", "w") as f:
        for i in range(len(file_names)):
            f.write(file_names[i].split("/")[-1].split(".")[0] + " [mm]  :  " + f"{pos_repeats[i]:.5f}" + "\n")
                
                
    