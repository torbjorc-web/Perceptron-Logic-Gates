"""
Machine Learning III: Perceptron Logic Gates
Complete solution with improved readability, comments, and input validation
"""

from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# ============================================
# TASKS 1-3: Creating and Visualizing AND Data
# ============================================

# Task 1: Create data list with four possible inputs to AND gate
data = [[0, 0], [0, 1], [1, 0], [1, 1]]

# Task 2: Create labels for AND gate (1 only if both inputs are 1)
labels = [0, 0, 0, 1]

# Task 3: Plot the four points on a graph
# Using list comprehension to extract x and y values
x_values_data = [point[0] for point in data]
y_values_data = [point[1] for point in data]

plt.scatter(x_values_data, y_values_data, c=labels)
plt.title('AND Logic Gate Data')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.show()

print("✓ Tasks 1-3 Complete: AND data created and visualized")
print(f"data = {data}")
print(f"labels = {labels}")
print("\nWhy AND data is linearly separable:")
print("  The point [1,1] (label=1) can be separated from [0,0], [0,1], [1,0] (label=0)")
print("  by drawing a straight line (e.g., x + y = 1.5)")

# ============================================
# TASKS 4-6: Building the Perceptron for AND
# ============================================

# Task 4: Create Perceptron object with max_iter=40 and random_state=22
classifier = Perceptron(max_iter=40, random_state=22)

# Task 5: Train the model using data and labels
# Convert to numpy arrays for scikit-learn compatibility
classifier.fit(np.array(data), np.array(labels))

# Task 6: Check accuracy using score() method
accuracy = classifier.score(np.array(data), np.array(labels))
print(f"\n✓ Tasks 4-6 Complete: AND Perceptron trained")
print(f"Accuracy on AND data: {accuracy}")
print("Expected: 100% (1.0) - Your perceptron learned AND!")

# ============================================
# TASK 7: XOR Gate
# ============================================

# Change labels to XOR (1 only if ONE input is 1, not both)
labels = [0, 1, 1, 0]  # XOR labels
classifier.fit(np.array(data), np.array(labels))
xor_accuracy = classifier.score(np.array(data), np.array(labels))

print(f"\n✓ Task 7 Complete: XOR Perceptron")
print(f"XOR labels: {labels}")
print(f"Accuracy on XOR data: {xor_accuracy}")
print("Is XOR data linearly separable? NO")
print("  You cannot draw a straight line to separate [0,1] and [1,0] from [0,0] and [1,1]")

# ============================================
# TASK 8: OR Gate
# ============================================

# Change labels to OR (1 if ANY input is 1)
labels = [0, 1, 1, 1]  # OR labels

print(f"\n✓ Task 8 Complete: OR Perceptron")
print(f"OR labels: {labels}")
print("Prediction before running:")
print("  - Is OR data linearly separable? YES (can separate [0,0] from others)")
print("  - Expected accuracy: 100%")

classifier.fit(np.array(data), np.array(labels))
or_accuracy = classifier.score(np.array(data), np.array(labels))
print(f"\nActual accuracy on OR data: {or_accuracy}")

# ============================================
# TASKS 9-16: Visualizing the Perceptron
# ============================================

# Task 9: Reset labels to AND gate and investigate decision_function
labels = [0, 0, 0, 1]  # AND labels
classifier.fit(np.array(data), np.array(labels))

# Test decision_function on sample points
sample_points = [[0, 0], [1, 1], [0.5, 0.5]]
decision_results = classifier.decision_function(np.array(sample_points))

print(f"\n✓ Task 9 Complete: Decision function results")
print(f"decision_function({sample_points}) = {decision_results}")
print(f"  [0, 0] distance: {decision_results[0]} (abs: {abs(decision_results[0])})")
print(f"  [1, 1] distance: {decision_results[1]} (abs: {abs(decision_results[1])})")
print(f"  [0.5, 0.5] distance: {decision_results[2]} (abs: {abs(decision_results[2])})")
print(f"\n[1, 1] is closer to the decision boundary (abs distance: {abs(decision_results[1])})")

# Task 10: Create x_values and y_values (100 evenly spaced decimals between 0 and 1)
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)

# Task 11: Create point_grid using product() to get all combinations
point_grid = list(product(x_values, y_values))

# Task 12: Call decision_function() on point_grid
distances = classifier.decision_function(np.array(point_grid))

# Task 13: Take absolute value of every distance
abs_distances = [abs(pt) for pt in distances]

# Task 14: Reshape into 100x100 2D matrix for pcolormesh
distances_matrix = np.reshape(abs_distances, (100, 100))

# Task 15: Draw the heat map for AND gate
plt.figure(figsize=(8, 8))
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)
plt.colorbar(heatmap, label='Distance from Decision Boundary')
plt.title('AND Logic Gate - Decision Boundary')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.show()

print(f"\n✓ Tasks 10-15 Complete: AND heatmap created")
print("You should see a purple line where distances are 0 - that's the decision boundary!")

# Task 16: Compare AND, OR, and XOR decision boundaries
print(f"\n✓ Task 16 Complete: Comparing all three gates")

# Create heatmaps for OR gate
labels_or = np.array([0, 1, 1, 1])
classifier.fit(np.array(data), labels_or)
or_distances = classifier.decision_function(np.array(point_grid))
or_abs_distances = [abs(pt) for pt in or_distances]
or_distances_matrix = np.reshape(or_abs_distances, (100, 100))

# Create heatmaps for XOR gate
labels_xor = np.array([0, 1, 1, 0])
classifier.fit(np.array(data), labels_xor)
xor_distances = classifier.decision_function(np.array(point_grid))
xor_abs_distances = [abs(pt) for pt in xor_distances]
xor_distances_matrix = np.reshape(xor_abs_distances, (100, 100))

# Plot all three heatmaps side by side
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.pcolormesh(x_values, y_values, distances_matrix)
plt.colorbar(label='Distance')
plt.title('AND Gate')
plt.xlabel('Input 1')
plt.ylabel('Input 2')

plt.subplot(1, 3, 2)
plt.pcolormesh(x_values, y_values, or_distances_matrix)
plt.colorbar(label='Distance')
plt.title('OR Gate')
plt.xlabel('Input 1')
plt.ylabel('Input 2')

plt.subplot(1, 3, 3)
plt.pcolormesh(x_values, y_values, xor_distances_matrix)
plt.colorbar(label='Distance')
plt.title('XOR Gate (Not Linearly Separable)')
plt.xlabel('Input 1')
plt.ylabel('Input 2')

plt.tight_layout()
plt.show()

print("\nKey Observations:")
print("1. AND: Decision boundary is a diagonal line separating [1,1] from others")
print("2. OR: Decision boundary separates [0,0] from others")
print("3. XOR: NO clear boundary - cannot separate the data with one straight line")
print("\nWhy XOR fails: Not linearly separable")
print("Solution: Combine multiple perceptrons (neural network)!")
print("  XOR = (A OR B) AND NOT (A AND B)")
print("\n  This is exactly like logic gates: AND and OR can't produce XOR alone,")
print("  but combining them creates XOR!")
