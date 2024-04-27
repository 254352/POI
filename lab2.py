import numpy as np
import csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np
import pyransac3d as pyrsc

#otwieranie pliku xyz
def points_reader():
    with open('points.xyz', newline='') as xyzfile:
        reader = csv.reader(xyzfile, delimiter=',')
        for x, y, z in reader:
            yield (float(x), float(y), float(z))

points_xyz =[]
for p in points_reader():
    points_xyz.append(p)

clusterer = KMeans(n_clusters=3)
X = np.array(points_xyz)
y_pred = clusterer.fit_predict(X)
red = y_pred == 0
blue = y_pred == 1
green = y_pred == 2

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

plt.scatter(X[red, 0], X[red, 1], X[red, 2], c="r")
plt.scatter(X[blue, 0], X[blue, 1], X[blue, 2], c="b")
plt.scatter(X[green, 0], X[green, 1], X[green, 2], c="g")
plt.show()

###########################################################

y_pred1 = DBSCAN(eps=0.3, min_samples=10).fit(X)

# Identify unique clusters
unique_labels = np.unique(y_pred)

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Assign colors to clusters and noise
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

# Plot each cluster with a unique color
for label, color in zip(unique_labels, colors):
    if label == -1:  # -1 indicates noise
        cluster_label = "Noise"
        marker = 'x'  # Different marker for noise
    else:
        cluster_label = f"Cluster {label}"
        marker = 'o'  # Default marker for clusters

    mask = (y_pred == label)
    ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], c=[color], label=cluster_label, marker=marker)

plt.legend()
plt.show()

###################################################################################################

plane1 = pyrsc.Plane()
best_eq_red, best_inliers_red = plane1.fit(X[red,:], 0.01)
print(best_eq_red)

plane2 = pyrsc.Plane()
best_eq_blue, best_inliers_blue = plane2.fit(X[blue,:], 0.01)
print(best_eq_blue)

plane3 = pyrsc.Plane()
best_eq_green, best_inliers_green = plane3.fit(X[green,:], 0.01)
print(best_eq_green)
