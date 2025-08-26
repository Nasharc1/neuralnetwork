import numpy as np
import matplotlib.pyplot as plt

# Read the file manually
def read_training_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # Split the line and convert to float
            try:
                values = [float(x) for x in line.strip().split()]
                if len(values) == 3:
                    data.append(values)
            except ValueError:
                # Skip lines that can't be converted
                continue
    return np.array(data)

# Load training data
full_training_data = read_training_data('project_training_data.dat')

print("Full training data shape:", full_training_data.shape)
print("Full training data contents:\n", full_training_data)

# Load network grid output
grid_output = np.loadtxt('network_grid_output.txt')

plt.figure(figsize=(12, 10))

# Separate positive and negative classes
pos_data = full_training_data[full_training_data[:, 2] == 1]
neg_data = full_training_data[full_training_data[:, 2] == -1]

print("\nPositive data points:")
print(pos_data)
print("\nNegative data points:")
print(neg_data)

# Plot training points
plt.scatter(pos_data[:, 0], pos_data[:, 1], c='red', marker='o', label='Positive Class (+1)', s=150)
plt.scatter(neg_data[:, 0], neg_data[:, 1], c='blue', marker='x', label='Negative Class (-1)', s=150)

# Create grid
x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)
X1, X2 = np.meshgrid(x1, x2)

# Reshape grid outputs
Z = grid_output.reshape(X1.shape)

# Plot decision boundary
plt.contour(X1, X2, Z, levels=[0], colors='green', linestyles='--', linewidths=2)

plt.title('Neural Network Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.tight_layout()
plt.savefig('decision_boundary.png')
plt.close()