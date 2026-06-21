Machine Learning III: Perceptron Logic Gates
Project Overview
This project uses a perceptron to model simple logic gates: AND, OR, and XOR. It shows how a perceptron can learn linearly separable data, and why it fails on XOR because XOR is not linearly separable.

What the Project Does
Creates the four possible input pairs for a 2-input logic gate.

Labels the points for AND, OR, and XOR.

Trains a perceptron on the data.

Measures accuracy on the same inputs.

Visualizes the decision boundary using a heat map.

Concepts Covered
Linearly separable data

Perceptrons

Decision functions

Decision boundaries

Heat map visualization

Results
AND gate: The perceptron learns AND successfully with 100% accuracy.

OR gate: The perceptron also learns OR successfully with 100% accuracy.

XOR gate: The perceptron fails to learn XOR because the data is not linearly separable.

Why XOR Fails
A single perceptron can only draw one straight decision boundary. XOR needs a non-linear boundary, so a single perceptron cannot solve it.

Files
main.py or notebook file: Contains the perceptron logic gate code.

This README: Explains the project.

How to Run
Open the project in Codecademy or your Python environment.

Run the code cells or script from top to bottom.

View the printed accuracy results and the plotted heat maps.

Example Code Used
python
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
Summary
This project demonstrates the basics of binary classification with perceptrons and shows the limitation of perceptrons on non-linearly separable data like XOR.
