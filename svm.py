import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt

cechy = pd.read_csv('wyniki_cech.csv', sep=',')

#======================Wizualizacja========================

data = np.array(cechy)
X = (data[:,:-1]).astype('float64')
Y = data[:,-1]

x_transform = PCA(n_components=3)

Xt = x_transform.fit_transform(X)

red = Y == 'beton'
blue = Y == 'drewno'
cyan = Y == 'marmur'

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(Xt[red, 0], Xt[red, 1], Xt[red, 2], c ="r")
ax.scatter(Xt[blue, 0], Xt[blue, 1], Xt[blue, 2], c ="b")
ax.scatter(Xt[cyan, 0], Xt[cyan, 1], Xt[cyan, 2], c ="c")
plt.show()

#Klasyfikacja

classifier = svm.SVC(gamma='auto')

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(acc)

cm= confusion_matrix(y_test, y_pred, labels=classifier.classes_, normalize='true')
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

