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
    
    
    
def wsd_punkte(data):
    """
    Standard deviation of the 3D points
    """
    # Extract x, y, z
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]
    # Calculate the standard deviation for each dimension
    std_x = np.std(xs, ddof=1)
    std_y = np.std(ys, ddof=1)
    std_z = np.std(zs, ddof=1)
    # Calculate the 3D standard deviation
    std_3d = np.sqrt(std_x**2 + std_y**2 + std_z**2)
    
    return std_3d
    
def wsd_vektoren(data):
    """
    Standard deviation of the orientation deviation of the vectors
    """
    # Calculate the mean vector
    mean_vector = np.mean(data, axis=0)
    # Calculate the orientation deviation of each vector to the mean vector
    d_phis = []
    for vector in data:
        d_phi = np.arccos(np.dot(mean_vector, vector) / (np.linalg.norm(mean_vector) * np.linalg.norm(vector)))
        d_phis.append(d_phi**2)
    # Calculate the standard deviation of the orientation deviation
    std_phi = np.sqrt(np.sum(d_phis) / (data.shape[0] - 1))
    
    return std_phi
    

def planes(data):
    """
    Calculate the centers and normal vectors of the planes defined by 3 corresponding points
    """
    # Initialize the arrays for the centers and normal vectors
    centers = np.zeros((data.shape[0]//3, 3))
    normal_vectors = np.zeros((data.shape[0]//3, 3))
    # Calculate the center and normal vector for each plane
    for i in range(data.shape[0]//3):
        p1 = data[i*3]
        p2 = data[i*3+1]
        p3 = data[i*3+2]
        # center
        center = (p1[1:4] + p2[1:4] + p3[1:4]) / 3
        centers[i] = center
        # normal vector
        v1 = p2[1:4] - p1[1:4]
        v2 = p3[1:4] - p1[1:4]
        normal_vector = np.cross(v1, v2)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        normal_vectors[i] = normal_vector
        
    return centers, normal_vectors

    
    
    
if __name__ == "__main__":
    # File names
    file_names = [
        "data/tracker/anfahrt_pose1.txt",
        "data/tracker/pose1.txt",
        "data/tracker/pose2.txt",
    ]
    
    # Choose the file
    file_name = file_names[1]
    
    # Read the file
    data = np.genfromtxt(file_name, delimiter=",", skip_header=0)
    
    # Get data by id
    target_1 = data[data[:, 0] == 1][:, 1:] # x, y, z
    target_2 = data[data[:, 0] == 2][:, 1:] # x, y, z
    target_3 = data[data[:, 0] == 3][:, 1:] # x, y, z
    
    # Calculate the standard deviation for each target
    target_1_std_3d = wsd_punkte(target_1)
    target_2_std_3d = wsd_punkte(target_2)
    target_3_std_3d = wsd_punkte(target_3)
    
    # Compute the centers and normal vectors of the planes defined by 3 corresponding points
    centers, normal_vectors = planes(data)
    
    # Calculate the standard deviation for the centers
    centers_std_3d = wsd_punkte(np.array(centers))
    
    # Calculate the standard deviation for the normal vectors
    normal_vectors_std = wsd_vektoren(np.array(normal_vectors))
    
    ### LOGGING ###
    print(f"Std. of target 1 [mm]: {target_1_std_3d:.3f}")
    print(f"Std. of target 2 [mm]: {target_2_std_3d:.3f}")
    print(f"Std. of target 3 [mm]: {target_3_std_3d:.3f}")
    print(f"Std. of the centers [mm]: {centers_std_3d:.3f}")
    print(f"Std. of the normal vectors [°]: {np.degrees(normal_vectors_std):.3f}")
    
    

    
    