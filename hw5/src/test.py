from sklearn.cluster import KMeans
import numpy as np
data = np.array([   [0.0,0.0,4.0],
                    [0.0,0.0,0.1],
                    [0.0,0.0,1.0],
                    [0.0,0.0,1.1],
                    [0.0,0.0,2.0],
                    [0.0,0.0,2.1],
                    [0.0,0.0,3.0],
                    [0.0,0.0,3.1]], dtype=np.float32)
sorted_data = sorted(data, key=lambda item: item[2])
print(sorted_data)
print(data)
# theta_values = data[:, 2].reshape(-1, 1)

# kmeans = KMeans(n_clusters=4)
# labels = kmeans.fit_predict(theta_values)
# centers = kmeans.cluster_centers_
# #print(labels)
# #print(centers)