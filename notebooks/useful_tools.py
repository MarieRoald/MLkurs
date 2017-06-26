'''A collection of useful functions'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools

from sklearn.metrics import confusion_matrix

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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    '''
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    Prints and plots the confusion matrix.
    classes: Name of classes
    normalize: Setting this to True will normalize the values
    '''
    with sns.axes_style('white'):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class ModelTester:
    '''A class for testing a model using the testing data, 
    i.e. printing MSE/success rate'''

    def __init__(self, x_test, y_test):
        '''
        x_test: Data related to the test set
        y_test: Response for the test set'''
        self.x_test = x_test
        self.y_test = y_test

    def _format_data(self, df, features):
        '''
        df: A dataframe, or a numpy array/matrix(TODO?) features: Iterable,
        containing the names of features to extract if df is a dataframe
        
        returns: If df is of type pandas.DataFrame, a new DataFrame (TODO: or
        view, rather?) containing the columns whose names are in features. If
        df is of another type , or noe features were specified, it must be
        assumed that the user has passed a valid data type of the correct
        dimensions to the method'''

        if type(df) == pd.core.frame.DataFrame and features != None:
            return df[features]

        return df

    def test_classifier(self, clf, class_names, features=None, title='Confusion matrix', 
            plot_confusion=True, print_success=True):

        predictions = clf.predict(self._format_data(self.x_test, features)).reshape((-1, 1))
        cnf_mat = confusion_matrix(self.y_test, predictions)

        if plot_confusion:
            plot_confusion_matrix(cnf_mat, class_names, title=title)
            plt.show()

        if print_success:
            print('Accuracy: ', (sum(np.equal(predictions, self.y_test.values))/len(self.y_test))[0]*100, '%')


