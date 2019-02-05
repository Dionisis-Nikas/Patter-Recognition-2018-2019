import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm,tqdm_notebook

from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPClassifier

df3 = pd.read_csv('processed-data/BSAS-data/train.csv')

# Normalize the Dataset
X = np.array(df3.drop(['target', 'zip code'], axis=1))
X = preprocessing.scale(X, axis=1)
# Separate the correct answer y from the Dataset
y = np.array(df3['target'])

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=100)
clf = MLPClassifier(hidden_layer_sizes=(4,4), activation='logistic', solver='sgd', alpha=1e-5, learning_rate='adaptive', random_state=100, verbose=False)

accuracy = np.array([])

for train_index, test_index in tqdm_notebook(rkf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    accuracy = np.append(accuracy, [clf.score(X_test,y_test)])

print(np.amax(accuracy))
print(np.amin(accuracy))

layers = len(clf.coefs_)

for i in range(layers):
    weight_matrix_i = clf.coefs_[i]
    print ('>>> Weight Matrix Layer ', i+1, '\n')
    tmp = pd.DataFrame(weight_matrix_i, columns=range(weight_matrix_i.shape[1]))
    print (tmp, '\n\n')