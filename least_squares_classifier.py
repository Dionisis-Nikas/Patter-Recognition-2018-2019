import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm,tqdm_notebook

from sklearn import preprocessing
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.linear_model import LinearRegression

df3 = pd.read_csv('processed-data/BSAS-data/train.csv')

# Normalize the Dataset
X = np.array(df3.drop(['target', 'zip code'], axis=1))
X = preprocessing.scale(X, axis=1)
# Separate the correct answer y from the Dataset
y = np.array(df3['target'])

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=100)

clf = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=-1)
accuracy = np.array([])

for train_index, test_index in tqdm_notebook(rkf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    accuracy = np.append(accuracy, [clf.score(X_test,y_test)])


print(np.amax(accuracy))
print(np.amin(accuracy))

for feature, coef in zip(np.delete(df3.columns, [2, 4]), clf.coef_):
    print (feature,"\t\tCoefficient in LS Line: ",coef)