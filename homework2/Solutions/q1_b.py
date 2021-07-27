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
data_train , data_test = train_test_split(
    data,
    #data.drop(labels=['Y'], axis=1),
    #data['Y'],
    train_size=500,
    shuffle=False)


C = np.linspace(0.0001,0.6,100)

import sklearn.model_selection as cv
#train = cv.KFold(n_splits=5)

cross_train = []
cross_test = []

for i in range(10):
    datatmp = data_train
    for rows in range(0 + 50*i, 50+50*i):
        datatmp = datatmp.drop(rows,axis=0)
    cross_test.append(data_train.loc[0 + 50 * i: 49 + 50 * i])
    cross_train.append(datatmp)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
scores = []
losses = []
mean_scores = []
best_c = 0

for c in C:
    score = []
    classifier = LogisticRegression(solver='liblinear', C=c, penalty='l1')
    loss = []
    values = 0
    for i in range(10):
        train_x = cross_train[i].drop(labels=['Y'], axis=1)
        train_y = cross_train[i]['Y']
        test_x = cross_test[i].drop(labels=['Y'], axis=1)
        test_y = cross_test[i]['Y']
        classifier.fit(train_x, train_y)
        
        pre_y = classifier.predict(test_x)
        values += pre_y
        pro_y = classifier.predict_proba(test_x)
        #print(pre_y)
        loss.append(metrics.log_loss(test_y, pro_y))
    losses.append(loss)
    #score.append(classifier.score(test_x, test_y))
    mean_score = np.mean(loss)
    mean_scores.append(mean_score)


import matplotlib.pyplot as plt
#scores = pd.DataFrame(scores)
#print(scores.values)
#print(clabels)
#print(losses)
mydata = {}
for i in range(0,100):
    mydata['C' + str(i)] = losses[i]
    #plt.boxplot(scores.values[i], labels= string)

df = pd.DataFrame(mydata)
df.plot.box(title="Different c choice")
plt.xlabel("Parameter number")
plt.ylabel("CV Performance")
plt.show()

#print(mean_scores)
cmax = mean_scores.index(min(mean_scores))
print("my choice of C:", C[cmax])
score_test = 0
score_train = 0
data_test_x = data_test.drop(labels=['Y'], axis=1)
data_test_y = data_test['Y']
for i in range(10):
    classifier = LogisticRegression(solver='liblinear', C=C[cmax], penalty='l1')
    
    train_x = cross_train[i].drop(labels=['Y'], axis=1)
    train_y = cross_train[i]['Y']
    test_x = cross_test[i].drop(labels=['Y'], axis=1)
    test_y = cross_test[i]['Y']
    classifier.fit(train_x, train_y)
    score_test += classifier.score(test_x, test_y)
    score_train += classifier.score(data_test_x, data_test_y)

print(f"Test score: {score_test}  Train_score:{score_train}")