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
    
    
    
def wiederholstandardabweichung(data):
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]
    std_x = np.std(xs, ddof=1)
    std_y = np.std(ys, ddof=1)
    std_z = np.std(zs, ddof=1)
    
    std_3d = np.sqrt(std_x**2 + std_y**2 + std_z**2)
    
    return std_3d

def planes(data):
    centers = np.zeros((data.shape[0]//3, 3))
    normal_vectors = np.zeros((data.shape[0]//3, 3))
    
    
    for i in range(data.shape[0]//3):
        p1 = data[i*3]
        p2 = data[i*3+1]
        p3 = data[i*3+2]
        
        # center
        center = (p1[1:4] + p2[1:4] + p3[1:4]) / 3
        centers[i] = center
        
        # normal_orientation
        v1 = p2[1:4] - p1[1:4]
        v2 = p3[1:4] - p1[1:4]
        normal_vector = np.cross(v1, v2)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        normal_vectors[i] = normal_vector
        
        # [[array([ 2349.93232733, -2364.147673  ,  -459.076537  ]), array([ 12096.85300977, -15111.83981502,  -8677.61321344])],
        #  [array([ 2349.884701  , -2364.14594133,  -458.89453033]), array([ 12090.57462032, -15112.9615487 ,  -8688.72366373])],
        #  [array([ 2349.89249533, -2364.248086  ,  -459.09247633]), array([ 12102.11733821, -15112.84260145,  -8675.64745948])],
        #  [array([ 2349.90541467, -2364.22354933,  -459.09215667]), array([ 12102.7354267 , -15110.71182886,  -8676.44165691])],
        #  [array([ 2349.90231733, -2364.19953633,  -458.94642933]), array([ 12097.7537146 , -15110.7570168 ,  -8684.92183852])],
        #  [array([ 2350.04549467, -2364.209098  ,  -459.217773  ]), array([ 12108.70259552, -15110.72212727,  -8663.94072639])],
        #  [array([ 2350.08003033, -2364.17776667,  -459.19804233]), array([ 12107.57995549, -15110.87688403,  -8663.89674441])],
        #  [array([ 2349.909894  , -2364.14414567,  -459.048335  ]), array([ 12096.42327436, -15111.72838159,  -8681.28673772])],
        #  [array([ 2349.977415  , -2364.17735467,  -459.040821  ]), array([ 12095.12276476, -15110.22732167,  -8680.73588872])],
        #  [array([ 2349.65186967, -2364.16907067,  -458.88385367]), array([ 12092.79800213, -15107.91751275,  -8691.72870302])]]
        
    return centers, normal_vectors

    
    
    
if __name__ == "__main__":
    # File name
    file_name = "data/tracker/anfahrt_pose1.txt"  # id, x, y, z
    # Read the file
    data = np.genfromtxt(file_name, delimiter=",", skip_header=0)
    
    # Get data by id
    target_1 = data[data[:, 0] == 1][:, 1:] # x, y, z
    target_2 = data[data[:, 0] == 2][:, 1:] # x, y, z
    target_3 = data[data[:, 0] == 3][:, 1:] # x, y, z
    
    target_1_std_3d = wiederholstandardabweichung(target_1)
    target_2_std_3d = wiederholstandardabweichung(target_2)
    target_3_std_3d = wiederholstandardabweichung(target_3)
    
    centers, normal_vectors = planes(data)
    
    centers_std_3d = wiederholstandardabweichung(np.array(centers))
    
    # print(f"centers: {centers}")
    print(f"centers_std_3d: {centers_std_3d}")
    print(f"target_1_std_3d: {target_1_std_3d}")
    print(f"target_2_std_3d: {target_2_std_3d}")
    print(f"target_3_std_3d: {target_3_std_3d}")
    # print(f"normal_vectors: {normal_vectors}")    
    
    