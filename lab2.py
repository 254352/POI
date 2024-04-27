import numpy as np
import csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RANSACRegressor



'''
# Generujemy przykładową chmurę punktów z szumem
np.random.seed(0)
n_samples = 100
n_outliers = 10

# Płaszczyzna prawdziwa
X = np.random.normal(size=(n_samples, 3))
X[:, 2] = 3 * X[:, 0] + 2 * X[:, 1] + 1

# Dodajemy szum (outliers)
outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, 3))
X[:n_outliers] = outliers

# Dopasowujemy model RANSAC
ransac = RANSACRegressor(estimator=LinearRegression(), min_samples=3, residual_threshold=2.0, max_trials=100)
ransac.fit(X[:, :2], X[:, 2])

# Współczynniki płaszczyzny
a, b = ransac.estimator_.coef_
c = ransac.estimator_.intercept_

# Wizualizacja wyników
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Punkty oryginalne
ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='b', label='Points')

# Siatka dla płaszczyzny
x1 = np.linspace(min(X[:, 0]), max(X[:, 0]), 10)
x2 = np.linspace(min(X[:, 1]), max(X[:, 1]), 10)
X1, X2 = np.meshgrid(x1, x2)
Z = a * X1 + b * X2 + c

# Rysujemy płaszczyznę
ax.plot_surface(X1, X2, Z, alpha=0.5, color='g', label='RANSAC Plane')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Z')

plt.show()

print("Współczynniki płaszczyzny:")
print("a =", a)
print("b =", b)
print("c =", c)
'''
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
