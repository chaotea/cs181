import numpy as np
import matplotlib.pyplot as plt

mnist_pics = np.load("data/images.npy")  # Load MNIST
num_images, height, width = mnist_pics.shape  # Reshape mnist_pics to be a 2D numpy array
mnist_pics = np.reshape(mnist_pics, newshape=(num_images, height * width))

class KMeans(object):
    def __init__(self, K):
        self.K = K
        self.losses = []

    def euclidean_distance(self, x1, x2):
        if x1.shape != x2.shape:
            print("Dimension mismatch when calculating euclidean distances")
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X):
        # Generate initial clusters
        N = X.shape[0]
        D = X.shape[1]
        clusters_index = []
        while len(clusters_index) < self.K:
            rand_index = np.random.randint(0, N)
            if rand_index not in clusters_index:
                clusters_index.append(rand_index)
        clusters = np.array([X[i] for i in clusters_index])
        
        # Iniitialize cluster assignments
        cluster_assignments = np.empty(N, dtype=int)
        
        # Clear losses
        self.losses = []
        
        # Begin fitting
        while True:
            # Assign new clusters
            new_cluster_assignments = np.empty(N, dtype=int)
            for index in range(N):
                current = X[index]
                min_cluster = None
                min_dist = float('inf')
                for i, c in enumerate(clusters):
                    dist = self.euclidean_distance(current, c)
                    if dist < min_dist:
                        min_dist = dist
                        min_cluster = i
                new_cluster_assignments[index] = min_cluster
            if (new_cluster_assignments==cluster_assignments).all():
                break
            cluster_assignments = new_cluster_assignments
            
            # Update cluster centers
            totals = np.zeros((self.K, D))
            counts = np.zeros(self.K, dtype=int)
            for index in range(N):
                cluster = cluster_assignments[index]
                totals[cluster] += X[index]
                counts[cluster] += 1
            new_clusters = []
            for j in range(self.K):
                new_clusters.append(totals[j] / counts[j])
            clusters = new_clusters
            
            # Calculate loss
            current_loss = 0
            for index in range(N):
                current = X[index]
                cluster = clusters[cluster_assignments[index]]
                temp = current - cluster
                current_loss += np.square(np.linalg.norm(temp))
            self.losses.append(current_loss / N)
        
        self.clusters = clusters
        self.cluster_assignments = cluster_assignments
        
    def get_mean_images(self):
        return self.clusters
    
    def plot_loss(self):
        plt.figure()
        plt.plot(self.losses)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()

KMeansClassifier = KMeans(K=10)
KMeansClassifier.fit(mnist_pics)
clusters = KMeansClassifier.get_mean_images()
print(f'Final loss: {KMeansClassifier.losses[-1]}')

plt.figure(figsize=(15,20))
plt.subplot(4, 3, 1)
for i in range(10):
    plt.subplot(4, 3, i+1)
    plt.imshow(clusters[i].reshape(28,28), cmap='Greys_r')
    plt.title(f'Cluster {i+1}')
plt.savefig('2.2 KMeans.png')