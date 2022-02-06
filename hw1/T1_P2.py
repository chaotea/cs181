#####################
# CS 181, Spring 2021
# Homework 1, Problem 2
# Start Code
##################

import math
import matplotlib.cm as cm

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

# Read from file and extract X and y
df = pd.read_csv('data/p2.csv')

X_df = df[['x1', 'x2']]
y_df = df['y']

X = X_df.values
y = y_df.values

print("y is:")
print(y)

def predict_kernel(alpha=0.1):
    """Returns predictions using kernel-based predictor with the specified alpha."""
    
    W = alpha * np.array([[1., 0.], [0., 1.]])

    def K(x1, x2, W):
        return np.exp(-1 * np.matmul(np.matmul(np.transpose(x1-x2), W), (x1-x2)))

    def f(x, W):
        numerator = sum(K(row[:2], x, W) * row[2] for row in df.values if not np.array_equal(row[:2], x))
        denominator = sum(K(row[:2], x, W) for row in df.values if not np.array_equal(row[:2], x))
        return numerator / denominator
    
    prediction = [f(x, W) for x in df[df.columns[:2]].values]
    return np.array(prediction)

def predict_knn(k=1):
    """Returns predictions using KNN predictor with the specified k."""

    W = np.array([[1., 0.], [0., 1.]])
    
    def K(x1, x2):
        return np.exp(-1 * np.matmul(np.matmul(np.transpose(x1-x2), W), (x1-x2)))
    
    # Compute distances and store in matrix
    n = len(df)
    distances = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            distances[i][j] = K(df.iloc[i][:2], df.iloc[j][:2])

    def get_neighbors(x, i):
        neighbors = []
        for j, distance in enumerate(distances[i]):
            if distance != 1:
                neighbors.append((j, distance))
        neighbors.sort(reverse=True, key=lambda x: x[1])
        neighbors = neighbors[:k]
        return [index for index, _ in neighbors]
    
    def f(x, i):
        sum_f = 0
        neighbors = get_neighbors(x, i)
        for j, y in enumerate(df['y']):
            if j in neighbors:
                sum_f += y
        return sum_f / k
    
    prediction = [f(x, i) for i, x in enumerate(df[['x1', 'x2']].values)]
    return np.array(prediction)

def plot_kernel_preds(alpha):
    title = 'Kernel Predictions with alpha = ' + str(alpha)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_kernel(alpha)
    print(y_pred)
    loss = 'L2: ' + str(sum((y - y_pred) ** 2))
    plt.text(0.50, 0.95, loss, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')
    for x_1, x_2, y_ in zip(df['x1'].values, df['x2'].values, y_pred):
        plt.annotate(str(round(y_, 2)),
                     (x_1, x_2), 
                     textcoords='offset points',
                     xytext=(0,5),
                     ha='center') 

    # Saving the image to a file, and showing it as well
    plt.savefig('alpha' + str(alpha) + '.png')
    plt.show()

def plot_knn_preds(k):
    title = 'KNN Predictions with k = ' + str(k)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_knn(k)
    print(y_pred)
    loss = 'L2: ' + str(sum((y - y_pred) ** 2))
    plt.text(0.50, 0.95, loss, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')
    for x_1, x_2, y_ in zip(df['x1'].values, df['x2'].values, y_pred):
        plt.annotate(str(round(y_, 2)),
                     (x_1, x_2), 
                     textcoords='offset points',
                     xytext=(0,5),
                     ha='center') 
    # Saving the image to a file, and showing it as well
    plt.savefig('k' + str(k) + '.png')
    plt.show()

for alpha in (0.1, 3, 10):
    # TODO: Print the loss for each chart.
    plot_kernel_preds(alpha)

for k in (1, 5, len(X)-1):
    # TODO: Print the loss for each chart.
    plot_knn_preds(k)