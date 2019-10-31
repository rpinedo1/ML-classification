# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# # Implementing a perceptron learning algorithm in Python
# ## An object-oriented perceptron API

class Perceptron(object):
    """Perceptron classifier.
    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.
    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.
        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        
        print("#" * 20)
        print(X[:10])
        print("#" * 20)

        for _ in range(self.n_iter):
            errors = 0
            print("NEW EPOCH")
            for xi, target in zip(X, y):
                print("this is xi: {0}".format(xi))
                prediction = self.predict(xi)
                print("prediction: {0}, actual: {1}".format(prediction, target))
                update = self.eta * (target - prediction)
                print("update: {0}".format(update))
                print("before update: ")
                print(self.w_)
                self.w_[1:] += update * xi
                print("after update")
                print(self.w_)
                print("before base unit: {0}".format(self.w_[0]))
                self.w_[0] += update
                print("after base unit: {0}".format(self.w_[0]))
                print("")
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, xi):
        """Calculate net input (z)"""
        """ z = w^(T).x = [w1, w2].[x1_1, x1_2] = w1*x1_1 + w2*x1_2"""

        z = np.dot(xi, self.w_[1:]) + self.w_[0]
        print("dot product:")
        print(z)
        return z

    def predict(self, xi):
        """Return class label after unit step"""
        return np.where(self.net_input(xi) >= 0.0, 1, -1)


v1 = np.array([1, 2, 3])
v2 = 0.5 * v1
print(np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
print(df.tail())

# tail of df looks like this:
#        0    1    2    3               4
# 145  6.7  3.0  5.2  2.3  Iris-virginica
# 146  6.3  2.5  5.0  1.9  Iris-virginica
# 147  6.5  3.0  5.2  2.0  Iris-virginica
# 148  6.2  3.4  5.4  2.3  Iris-virginica
# 149  5.9  3.0  5.1  1.8  Iris-virginica

# select setosa and versicolor from df (iris dataset, first 100 samples, 50=setosa & 50=versicolor)
# 4th column b/c that's where the names are 
y = df.iloc[0:100,4].values
# iris-setosa will be (-1) the other will be (+1)
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
# X is a matrix with the values of sepal (column 0) and petal (column 2) lengths
# we skipped column 1 
X = df.iloc[0:100, [0,2]].values

# plot data 
plt.scatter(X[:50,0], X[:50,1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100,0], X[50:100,1], color='blue', marker='x', label='versicolor')

plt.ylabel('sepal length [cm]')
plt.xlabel('petal length [cm]')

plt.legend(loc='upper left')

plt.show()


# TRAINING THE PERCEPTRON
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_)+1), ppn.errors_, marker='o')

plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()
