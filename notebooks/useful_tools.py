'''A collection of useful functions'''
import numpy as np
import matplotlib.pyplot as plt

def plot_boundary(X, clf, plot_step=0.02, padding=0.1):
    '''Plots the decision boundary in two dimensions for a classifier
    
    X: Data
    clf: Classifier
    plot_step: Distance between each point in the mesh
    padding: Padding in each direction'''
    
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
