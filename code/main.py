import numpy as np  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


### PLOTTING FUNCTIONS ###
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

def plot_all(points, centers, normal_vectors):
    """
    Plot the 3D points, the centers of the planes, the normal vectors of the planes, 
    and a triangle through Target 1, Target 2, and Target 3.
    """
    
    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define colors and labels for three targets
    colors = ['r', 'g', 'b']
    labels = ['Target 1', 'Target 2', 'Target 3']
    
    # Scatter plot for the three targets
    for i in range(1, 4):
        subset = points[points[:, 0] == i]  # Select points for target i
        ax.scatter(subset[:, 1], subset[:, 2], subset[:, 3], c=colors[i-1], label=labels[i-1], alpha=0.7)

    # Scatter plot for centers (brown)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='brown', label='Centers', alpha=0.9, marker='o', s=50)

    # Plot normal vectors (black arrows)
    for i in range(centers.shape[0]):
        ax.quiver(
            centers[i, 0], centers[i, 1], centers[i, 2], 
            normal_vectors[i, 0], normal_vectors[i, 1], normal_vectors[i, 2], 
            color='k', length=50, normalize=True
        )

    # Draw triangles for each set of target positions
    num_triangles = min(len(points) // 3, 10)  # Ensure there are enough points

    for j in range(num_triangles):
        target_points = []
        for i in range(1, 4):  # Collect one point for each target (1, 2, 3)
            subset = points[points[:, 0] == i]
            if subset.shape[0] > j:  # Ensure there are enough points
                target_points.append(subset[j, 1:4])  # Take the j-th point for each target
        
        if len(target_points) == 3:
            triangle = np.array(target_points)
            ax.add_collection3d(Poly3DCollection([triangle], alpha=0.3, color='cyan', edgecolor='k'))

    
    # Set equal axis scale
    def set_axes_equal(ax):
        """Make the 3D plot axes have equal scale."""
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        z_limits = ax.get_zlim()

        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        z_range = z_limits[1] - z_limits[0]

        max_range = max(x_range, y_range, z_range)

        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        ax.set_xlim([x_middle - max_range / 2, x_middle + max_range / 2])
        ax.set_ylim([y_middle - max_range / 2, y_middle + max_range / 2])
        ax.set_zlim([z_middle - max_range / 2, z_middle + max_range / 2])

    set_axes_equal(ax)

    # Labels and title
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("3D Plot of Target Positions")
    ax.legend()
    
    plt.show()

    
### EVALUATION FUNCTIONS ###    
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
    
def orient_repeat(v):
    """
    Orientation repeatability of the normal vectors
    """
    # Mmagnitude of each vector
    v_mag = np.linalg.norm(v, axis=1)
    # Angles with each axis in degrees
    a = np.degrees(np.arccos(v[:, 0] / v_mag))
    b = np.degrees(np.arccos(v[:, 1] / v_mag))
    c = np.degrees(np.arccos(v[:, 2] / v_mag))
    
    # Standard deviation of the angles
    std_a = np.std(a, ddof=1)
    std_b = np.std(b, ddof=1)
    std_c = np.std(c, ddof=1)
    
    # 3 sigma repeatability
    RPa = 3 * std_a
    RPb = 3 * std_b
    RPc = 3 * std_c
    
    return RPa, RPb, RPc
    
def normal_vectors(data):
    """
    Calculate the normal vectors of the (triangular) planes defined by 3 corresponding points
    """
    # Initialize the array for the normal vectors
    normal_vectors = np.zeros((data.shape[0]//3, 3))
    # Calculate the center and normal vector for each plane
    for i in range(data.shape[0]//3):
        # 3 points
        p1 = data[i*3]
        p2 = data[i*3+1]
        p3 = data[i*3+2]
        # normal vector
        v1 = p2[1:4] - p1[1:4]
        v2 = p3[1:4] - p1[1:4]
        normal_vector = np.cross(v1, v2)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        normal_vectors[i] = normal_vector
        
    return normal_vectors

    
if __name__ == "__main__":   
    # File names
    file_names = [
        "data/tracker/pose1.txt",
        "data/tracker/pose2.txt",
        "data/tracker/anfahrt_pose1.txt",
    ]
    
    for i, file_name in enumerate(file_names):
        print(f"{i+1}: {file_name}")
    
        # Read the file
        data = np.genfromtxt(file_name, delimiter=",", skip_header=0)
                
        # Data for each target
        target_1 = data[data[:, 0] == 1][:, 1:] # x, y, z
        target_2 = data[data[:, 0] == 2][:, 1:] # x, y, z
        target_3 = data[data[:, 0] == 3][:, 1:] # x, y, z
        
        # Centers of the 3 corresponding target points
        centers = (target_1 + target_2 + target_3) / 3 # x, y, z
        
        # Positioning repeatability
        centers_3d_repeat   = pos_repeat(centers)
        target_1_pos_repeat = pos_repeat(target_1)
        target_2_pos_repeat = pos_repeat(target_2)
        target_3_pos_repeat = pos_repeat(target_3)
        
        # Standard deviation of the 3D points
        centers_std_3d  = wsd_punkte(centers)
        target_1_std_3d = wsd_punkte(target_1)
        target_2_std_3d = wsd_punkte(target_2)
        target_3_std_3d = wsd_punkte(target_3)
        
        # Normal vectors
        norm_vecs = normal_vectors(data)
        # Orientation repeatability
        RPa, RPb, RPc = orient_repeat(norm_vecs)
        
        ### PLOTTING ###
        # plot_raw(data)
        # plot_all(data, centers, normal_vectors)
        
        ### LOGGING ###
        print(f"Pos. repeat. all points      : {centers_3d_repeat:.4f} mm")
        print(f"Pos. repeat.  targ. 1        : {target_1_pos_repeat:.4f} mm")
        print(f"Pos. repeat.  targ. 2        : {target_2_pos_repeat:.4f} mm")
        print(f"Pos. repeat.  targ. 3        : {target_3_pos_repeat:.4f} mm")
        print("-----------------------------------------")
        print(f"Std. of all points           : {centers_std_3d:.4f} mm")
        print(f"Std. of target 1             : {target_1_std_3d:.4f} mm")
        print(f"Std. of target 2             : {target_2_std_3d:.4f} mm")
        print(f"Std. of target 3             : {target_3_std_3d:.4f} mm")
        print("-----------------------------------------")
        print(f"Orient. repeat.  norm. vec. 1: {RPa:.4f}°")
        print(f"Orient. repeat.  norm. vec. 2: {RPb:.4f}°")
        print(f"Orient. repeat.  norm. vec. 3: {RPc:.4f}°")
        print("=="*50)
        
        ### WRITING TO FILE ###
        with open("data/results/" + file_name.split("/")[-1].split(".")[0] + "_results.txt", "w", encoding="utf-8") as f:
            f.write("#########################################\n")
            f.write("#     ISO 9283: 1998 (E) Evaluation     #\n")
            f.write("#########################################\n")
            f.write(f"Pos. repeat. all points       : {centers_3d_repeat:.4f} mm\n")
            f.write(f"Pos. repeat.  targ. 1         : {target_1_pos_repeat:.4f} mm\n")
            f.write(f"Pos. repeat.  targ. 2         : {target_2_pos_repeat:.4f} mm\n")
            f.write(f"Pos. repeat.  targ. 3         : {target_3_pos_repeat:.4f} mm\n")
            f.write("-----------------------------------------\n")
            f.write(f"Orient. repeat.  norm. vec. 1 : {RPa:.4f} °\n")
            f.write(f"Orient. repeat.  norm. vec. 2 : {RPb:.4f} °\n")
            f.write(f"Orient. repeat.  norm. vec. 3 : {RPc:.4f} °\n")
            f.write("-----------------------------------------\n")
            f.write("#########################################\n")
            f.write("#            Own Evaluation             #\n")
            f.write("#########################################\n")
            f.write(f"Std. of all points            : {centers_std_3d:.4f} mm\n")
            f.write(f"Std. of target 1              : {target_1_std_3d:.4f} mm\n")
            f.write(f"Std. of target 2              : {target_2_std_3d:.4f} mm\n")
            f.write(f"Std. of target 3              : {target_3_std_3d:.4f} mm\n")