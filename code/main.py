import matplotlib.pyplot as plt
import numpy as np

# File name
file_name = "anfahrt_pose1"

# Read the data
data = []
with open(file_name, "r") as file:
    for line in file:
        parts = line.strip().split(",")
        if len(parts) == 4:
            target_id = int(parts[0])
            x, y, z = map(float, parts[1:])
            data.append((target_id, x, y, z))

# Convert to numpy array
data = np.array(data)

# Separate by target IDs
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