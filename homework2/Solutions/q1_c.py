import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
# load dataset
data = pd.read_csv('Q1.csv')
data.head()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_vars]
data.shape
data_train,data_test = train_test_split(
    data,
    #data.drop(labels=['Y'], axis=1),
    #data['Y'],
    train_size=500,
    shuffle=False)

x_train = data_train.drop(labels=['Y'], axis=1)
y_train = data_train['Y']
x_test = data_test.drop(labels=['Y'], axis=1)
y_test = data_test['Y']

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = {'C': np.linspace(0.0001,0.6,100)}

grid_lr = GridSearchCV(estimator= LogisticRegression(penalty='l1', solver='liblinear'), 
                       param_grid=param_grid, cv=10)
grid_lr.fit(x_train, y_train)

print(f"Test score: {grid_lr.score(x_test, y_test)}")
print(f"Train score: {grid_lr.score(x_train, y_train)}")
print("best parameter: ", grid_lr.best_params_)


import sklearn.model_selection as cv
#skf = ShuffleSplit(n_splits=3, random_state=0)
skf = cv.KFold(n_splits=10)
grid_lr = GridSearchCV(estimator= LogisticRegression(penalty='l1', solver='liblinear'), 
                       param_grid=param_grid, cv=skf, scoring='neg_log_loss')
grid_lr.fit(x_train, y_train)
print()
print("best parameter: ", grid_lr.best_params_)


# from sklearn import metrics
# best_model = grid_lr.best_estimator_
# predict_y=best_model.predict(x_train)
classifier = LogisticRegression(solver='liblinear', C=grid_lr.best_params_['C'], penalty='l1')
classifier.fit(x_train, y_train)
print(f"Test score: {classifier.score(x_test, y_test)}")
print(f"Train score: {classifier.score(x_train, y_train)}")