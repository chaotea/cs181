import numpy as np
import matplotlib.pyplot as plt

mnist_pics = np.load("data/images.npy")  # Load MNIST
num_images, height, width = mnist_pics.shape  # Reshape mnist_pics to be a 2D numpy array
mnist_pics = np.reshape(mnist_pics, newshape=(num_images, height * width))

def get_cumul_var(mnist_pics, num_leading_components=500):
    # Standardize the input
    N = mnist_pics.shape[0]
    centered = (mnist_pics - np.mean(mnist_pics, axis=0))
    
    # Decompose the matrix
    u, s, vh = np.linalg.svd(centered)
    eigenvals = s ** 2 / N
    eigenvecs = vh
    
    # Calculate reconstruction error
    mean_error = np.sum(np.square(np.linalg.norm(centered, axis=1))) / N
    print(f'Reconstruction error using mean of dataset: {mean_error}')
    reconstructed = np.matmul(np.matmul(u[:,:10], np.diag(s[:10])), vh[:10])
    error = np.sum(np.square(np.linalg.norm(centered - reconstructed, axis=1))) / N
    print(f'Reconstruction error using first 10 principal components: {error}')
    
    # Compute cumulative proportion of variance explained
    var_props = eigenvals / np.sum(eigenvals)
    cumulative = [0] * len(var_props)
    cumulative[0] = var_props[0]
    for i in range(1, len(cumulative)):
        cumulative[i] = cumulative[i-1] + var_props[i]
    return eigenvals[:num_leading_components], eigenvecs[:num_leading_components].T, cumulative[:num_leading_components]

eigenvals, eigenvecs, cumulative = get_cumul_var(mnist_pics=mnist_pics, num_leading_components=500)

plt.style.use('seaborn')
plt.figure()
plt.scatter(np.linspace(0, 500, 500), eigenvals, s=8, c='tab:blue')
plt.title('Eigenvalues for 500 most significant components')
plt.savefig('2.1 eigenvalues.png')
plt.figure()
plt.plot(cumulative, c='tab:red')
plt.title('Cumulative proportion of variance explained')
plt.savefig('2.1 cumulative.png')

plt.style.use('default')
plt.figure(figsize=(15,20))
plt.subplot(4, 3, 1)
plt.imshow(np.mean(mnist_pics, axis=0).reshape(28,28), cmap='Greys_r')
plt.title('Mean image of dataset')
for i in range(10):
    plt.subplot(4, 3, i+2)
    plt.imshow(eigenvecs[:,i].reshape(28,28), cmap='Greys_r')
    plt.title(f'Mean image of component {i+1}')
plt.savefig('2.2 PCA.png')