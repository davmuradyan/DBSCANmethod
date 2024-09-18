import matplotlib.pyplot as plt
import numpy as np

def EuclideanDistance(pointA, pointB):
    return np.linalg.norm(pointA - pointB)
def DistanceMatrix(data):
    n = len(data)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            matrix[j][i] = EuclideanDistance(data[i], data[j])
    return matrix
def FindCentroid(distances, n, minPts, eps):
    if minPts <= 0:
        print("Error: minPts must be greater than or equal to zero.")
        return
    elif eps <= 0:
        print("Error: eps must be greater than or equal to zero.")
        return

    numOfNeighbours = [0 for i in range(n)]
    neighbours = [[] for i in range(n)]

    # Checking Horizontally (including 0)
    print(distances)
    for i in range(n):
        for j in range(i+1):
            if distances[i][j] <= eps:
                numOfNeighbours[i] += 1
                if i != j:
                    neighbours[i].append(j)

    # Checking Vertically (not included 0)
    for i in range(n):
        for j in range(i+1, n):
            if distances[j][i] <= eps:
                numOfNeighbours[i] += 1
                if i != j:
                    neighbours[i].append(j)

    # Creating array of centroids
    # If 1 -> centroid else -> noise
    centroids: list[int] = []
    for i in range(n):
        if numOfNeighbours[i] >= minPts:
            centroids.append(1)
        else:
            centroids.append(0)

    return centroids, neighbours
def Checker(point, clusters):
    point = point.tolist()
    print(point)
    for cluster in clusters:
        for p in cluster:
            if np.array_equal(point, p):
                return True
    return False
def DBSCAN(data, minPts, eps):
    n = len(data)
    distances = DistanceMatrix(data)

    centroids, neighbours = FindCentroid(distances, n, minPts, eps)

    clusters = []
    print(clusters)
    for i in range(n):
        if centroids[i] == 1 and not Checker(data[i], clusters):
            newCluster = [data[i]]
            for j in neighbours[i]:
                newCluster.append(data[j])
            clusters.append(newCluster)
    print(clusters)
    return clusters
def plot_clusters(data, clusters):
    if clusters is None or len(clusters) == 0:
        print("No clusters found.")
        return

    plt.figure(figsize=(8, 6))

    # Plot each cluster
    for i, cluster in enumerate(clusters):
        cluster_points = np.array(cluster)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')

    # Identify noise points
    all_cluster_points = np.concatenate(clusters) if len(clusters) > 0 else np.empty((0, data.shape[1]))
    noise_points = np.array([point for point in data if not any(np.array_equal(point, p) for p in all_cluster_points)])

    if len(noise_points) > 0:
        plt.scatter(noise_points[:, 0], noise_points[:, 1], color='gray', label='Noise')

    plt.title('DBSCAN Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

# Example usage
data = np.array([
    [1, 2],
    [2, 3],
    [6, 7],
    [8, 9]
])

minPts = 2
eps = 3.0

result_clusters = DBSCAN(data, minPts, eps)
if result_clusters is not None:
    for i, cluster in enumerate(result_clusters):
        print(f"Cluster {i}: {cluster}")
    plot_clusters(data, result_clusters)
else:
    print("No clusters found with the given parameters.")