import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax


# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam
        self.iterations = []
        self.loss = []

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, y):
        # Convert y to one-hot vectors
        C = np.eye(3)[y]
        # Add a column for bias to X
        X = np.column_stack((np.ones(X.shape[0]), X))  # X : 27 x 3
        # Create random initial W
        self.W = np.random.rand(X.shape[1], 3)  # W : 3 x 3

        for i in range(200000):
            wTx = np.matmul(X, self.W.T)
            softmaxed = softmax(wTx, axis=1)  # softmaxed : 27 x 3
            # Calculate loss every 1000 iterations
            if i % 1000 == 0:
                self.iterations.append(i)
                self.loss.append(-np.sum(np.log(softmaxed) * C))
            L = np.matmul((softmaxed - C).T, X) / X.shape[0]
            self.W -= self.eta * (L + self.lam * self.W)

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        X_pred = np.column_stack((np.ones(X_pred.shape[0]), X_pred))
        prediction = softmax(np.matmul(X_pred, self.W.T))
        return np.argmax(prediction, axis=1)

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        plt.figure()
        plt.plot(self.iterations, self.loss)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Negative Log-Likelihood Loss')
        plt.title(f'{output_file} eta {self.eta} lam {self.lam}')
        plt.savefig(f'{output_file} eta {self.eta} lam {self.lam}.png')
        if show_charts:
            plt.show()
