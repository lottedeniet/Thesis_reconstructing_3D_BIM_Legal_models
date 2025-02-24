import numpy as np
import matplotlib.pyplot as plt

# Reference rotation angles (rot + scale)
rotation_angles2 = np.array([171.3, 180, -43.5, 78.9, 6.8, 0.0, 121.6, 120, 22.2, 7])

# Rotation angles for different alpha values
# angles_alpha_0 = np.array([171.0, 171.0,  -49.0, 81.0, 16.0, 8.0, 122.5, 111.5, 18.5,  5.0])
# angles_alpha_0_1 = np.array([171.5, 171.5,  -49.0, 81.0, 16.0, 89.5, 122.5, 111.5, 18.5,  5.0])
# angles_alpha_0_2 = np.array([173.5, 173.5,  -49.0, 81.0, 16.0, 89.5, 122.5, 113.5, 18.5,  11.5])
# angles_alpha_0_5 = np.array([174.5, 174.5,  -44.5, 81.0, 16.0, 89.5, 122.5, 115.5, 18.5, 10.5])
# angles_alpha_0_8 = np.array([174.5, 174.5,  -44.5, 81.0, 16.0, 91.0, 123.5, 118.0, 15.0,  12.5])
# angles_alpha_1 = np.array([174.0, 174.0,  -44.5, 79.0, 17.0, 93.0, 123.0, 119.0, -157.5,  12.5])


# Rotation angles (only rot)
angles_alpha_0 =[(160.5), (160.5), (178.5), (-49.0), (78.5), (16.0), (10.0), (127.5), (113.5), (12.5), (12.5), (16.0)]
angles_alpha_0_1 =[(176.0), (176.0), (178.5), (-49.0), (78.5), (16.0), (10.0), (123.5), (113.5), (14.0), (14.0), (16.0)]
angles_alpha_0_2 =[(174.0), (174.0), (179.5), (-49.0), (78.5), (16.0), (10.0), (123.5), (113.5), (15.5), (15.5), (16.0)]
angles_alpha_0_5 =[(174.5), (174.5), (-178.5), (-44.5), (78.5), (16.0), (9.5), (123.5), (113.5), (17.0), (17.0), (15.5)]
angles_alpha_0_8 =[(174.5), (174.5), (-178.0), (-44.5), (78.5), (16.0), (89.0), (123.5), (118.0), (17.0), (17.0), (15.5)]
angles_alpha_1 =[(174.5), (174.5), (-177.5), (-44.5), (77.5), (17.0), (54.5), (123.5), (119.0), (-157.0), (-157.0), (14.5)]



# Function to compute circular difference between angles
def compute_circular_difference(reference, angles):
    min_len = min(len(reference), len(angles))
    return abs(((angles[:min_len] - reference[:min_len] + 180) % 360) - 180)

# Compute differences
diff_alpha_0 = compute_circular_difference(rotation_angles2, angles_alpha_0)
diff_alpha_0_1 = compute_circular_difference(rotation_angles2, angles_alpha_0_1)
diff_alpha_0_2 = compute_circular_difference(rotation_angles2, angles_alpha_0_2)
diff_alpha_0_5 = compute_circular_difference(rotation_angles2, angles_alpha_0_5)
diff_alpha_0_8 = compute_circular_difference(rotation_angles2, angles_alpha_0_8)
diff_alpha_1 = compute_circular_difference(rotation_angles2, angles_alpha_1)

# Create the plot
x_values = np.arange(len(diff_alpha_0))  # X-axis labels (index positions)

plt.figure(figsize=(12, 5))

plt.plot(x_values, diff_alpha_0, label='Alpha 0', color='purple', marker='o')
plt.plot(x_values, diff_alpha_0_1, label='Alpha 0.1', color='blue', marker='s')
plt.plot(x_values, diff_alpha_0_2, label='Alpha 0.2', color='green', marker='^')
plt.plot(x_values, diff_alpha_0_5, label='Alpha 0.5', color='yellow', marker='d')
plt.plot(x_values, diff_alpha_0_8, label='Alpha 0.8', color='orange', marker='v')
plt.plot(x_values, diff_alpha_1, label='Alpha 1', color='red', marker='x')

plt.axhline(0, color='black', linestyle='dotted', linewidth=1)  # Zero reference line

plt.xlabel("Index")
plt.ylabel("Angle Difference (Degrees)")
plt.title("Rotation Angle Difference (only rotation optimization)")
plt.legend()
plt.grid(True)

plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Reference rotation angles
rotation_angles2 = np.array([171.3, 180, -43.5, 78.9, 6.8, 0.0, 121.6, 120, 22.2, 7])

# Rotation angles for different alpha values
# angles_dict = {
#     "Alpha 0": np.array([171.0, 171.0,  -49.0, 81.0, 16.0, 8.0, 122.5, 111.5, 18.5,  5.0]),
#     "Alpha 0.1": np.array([171.5, 171.5,  -49.0, 81.0, 16.0, 89.5, 122.5, 111.5, 18.5, 5.0]),
#     "Alpha 0.2": np.array([173.5, 173.5,  -49.0, 81.0, 16.0, 89.5, 122.5, 113.5, 18.5, 11.5]),
#     "Alpha 0.5": np.array([174.5, 174.5, -44.5, 81.0, 16.0, 89.5, 122.5, 115.5, 18.5,  10.5]),
#     "Alpha 0.8": np.array([174.5, 174.5,  -44.5, 81.0, 16.0, 91.0, 123.5, 118.0, 15.0,  12.5]),
#     "Alpha 1": np.array([174.0, 174.0,  -44.5, 79.0, 17.0, 93.0, 123.0, 119.0, -157.5,  12.5]),
# }

angles_dict = {
    "Alpha 0": [(160.5),  (178.5), (-49.0), (78.5), (16.0), (10.0), (127.5), (113.5), (12.5),  (16.0)],
    "Alpha 0.1": [(176.0),  (178.5), (-49.0), (78.5), (16.0), (10.0), (123.5), (113.5), (14.0), (16.0)],
    "Alpha 0.2": [(174.0),  (179.5), (-49.0), (78.5), (16.0), (10.0), (123.5), (113.5), (15.5),  (16.0)],
    "Alpha 0.5": [(174.5),  (-178.5), (-44.5), (78.5), (16.0), (9.5), (123.5), (113.5), (17.0),  (15.5)],
    "Alpha 0.8": [(174.5),  (-178.0), (-44.5), (78.5), (16.0), (89.0), (123.5), (118.0), (17.0),  (15.5)],
    "Alpha 1":[(174.5),  (-177.5), (-44.5), (77.5), (17.0), (54.5), (123.5), (119.0), (-157.0),  (14.5)]
}
# Function to compute circular difference
def compute_circular_difference(reference, angles):
    min_len = min(len(reference), len(angles))
    return ((angles[:min_len] - reference[:min_len] + 180) % 360) - 180

# Compute average absolute differences
mean_differences = {
    alpha: np.mean(np.abs(compute_circular_difference(rotation_angles2, angles)))
    for alpha, angles in angles_dict.items()
}

# Create a bar chart
plt.figure(figsize=(8, 5))
plt.bar(mean_differences.keys(), mean_differences.values(), color=['purple', 'blue', 'green', 'yellow', 'orange', 'red'])

plt.xlabel("Alpha Value")
plt.ylabel("Mean Absolute Angle Difference (Degrees)")
plt.title("Comparison of Mean Absolute Angle Differences for Different Alpha Values")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
