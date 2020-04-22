import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
X, Y = fetch_openml('mnist_784', version=1, data_home='./scikit_learn_data', return_X_y=True)
X = X / 255.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)
from sklearn.svm import LinearSVC
svc=LinearSVC()
svc.fit(X_train, Y_train)
y_predtrain= svc.predict(X_train)
train_accuracy= np.mean(y_predtrain == Y_train)
y_predtest = svc.predict(X_test)
test_accuracy= np.mean(y_predtest ==Y_test)


print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))
