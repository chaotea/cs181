import numpy as np

data = [(0., 0., 0.),
        (0., 0.5, 0.),
        (0., 1., 0.),
        (0.5, 0., 0.5),
        (0.5, 0.5, 0.5),
        (0.5, 1., 0.5),
        (1., 0., 1.),
        (1., 0.5, 1.),
        (1., 1., 1.)]

alpha = 100

W1 = alpha * np.array([[1., 0.], [0., 1.]])
W2 = alpha * np.array([[0.1, 0.], [0., 1.]])
W3 = alpha * np.array([[1., 0.], [0., 0.1]])

data_arr = np.asarray(data)

def compute_loss(W):
    def k(x1, x2, W):
        return np.exp(-1 * np.matmul(np.matmul(np.transpose(x1-x2), W), (x1-x2)))

    def f(x, W):
        numerator = sum(k(row[:2], x, W) * row[2] for row in data_arr if not np.array_equal(row[:2], x))
        denominator = sum(k(row[:2], x, W) for row in data_arr if not np.array_equal(row[:2], x))
        return numerator / denominator
    
    return sum((row[2] - f(row[:2], W)) ** 2 for row in data_arr)


print(compute_loss(W1))
print(compute_loss(W2))
print(compute_loss(W3))