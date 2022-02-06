import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, y):
        self.mu = []
        self.shared_covariance = np.zeros((2,2))
        self.covariances = []

        for k in range(3):
            x_vals = X[y==k,:]
            self.mu.append(np.mean(x_vals, axis=0))
            if self.is_shared_covariance:
                self.shared_covariance += len(x_vals) / len(X) * np.cov(x_vals, rowvar=False)
            else:
                self.covariances.append(np.cov(x_vals, rowvar=False))
        return

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = []
        for k in range(3):
            covariance = self.shared_covariance if self.is_shared_covariance else self.covariances[k]
            preds.append(mvn.pdf(X_pred, self.mu[k], covariance))
        preds = np.array(preds)
        if preds.ndim == 1:
            preds = np.reshape(preds, (preds.shape[0], 1))
        return np.argmax(preds.T, axis=1)

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        ret = 0
        for k in range(3):
            covariance = self.shared_covariance if self.is_shared_covariance else self.covariances[k]
            ret += -1 * np.sum(mvn.logpdf(X[y==k], self.mu[k], covariance) + np.log(len(y[y==k]) / len(y)))
        return ret
