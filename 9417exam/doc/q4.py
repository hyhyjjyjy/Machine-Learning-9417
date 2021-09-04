import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


train_data = pd.read_csv('Q4_train.csv')
train_data = train_data.iloc[:,1:]
test_data = pd.read_csv('Q4_test.csv')
test_data = test_data.iloc[:,1:]

x_train = np.array(train_data.iloc[:,0:-1])
y_train = np.array(train_data.iloc[:,-1:])
x_test = np.array(test_data.iloc[:,0:-1])
y_test = np.array(test_data.iloc[:,-1:])


def total_loss(X, y, Z, models):
    '''
    computes total loss achieved on X, y based on linear regressions stored in models, and partitioning Z
    
    :param X: design matrix, n x p (np.array shape (n,p))
    :param y: response vector, n x 1 (np.array shape (n,1) or (n,))
    :param Z: assignment vector, n x 1 (assigns each sample to a partition)
    :param models: a list of M sklearn LinearRegression models, one for each of the partitions
    
    :returns: the loss of your complete model as computed in (a)
    '''

    loss = 0             
    M = len(models)
    n = len(Z)
    
    for m in range(0, M):
        
        data_x_m = []
        data_y_m = []
        for i in range(n):
            if (Z[i] == m):
                data_x_m.append(X[i])
                data_y_m.append(y[i]) 
        
        if (len(data_x_m) > 0):
            y_pre = models[m].predict(data_x_m)

            one_loss = 0
            for i in range(len(y_pre)):
                if (Z[i] == m):
                    one_loss += (y_pre[i] - data_y_m[i])**2
            loss += one_loss

    return loss

def find_partitions(X, y, models):
    '''
    given M models, assigns points in X to one of the M partitions
    
    :param X: design matrix, n x p (np.array shape (n,p))
    :param y: response vector, n x 1 (np.array shape (n,1) or (n,))
    :param models: a list of M sklearn LinearRegression models for each 
    of the partitions
    
    :returns: Z, a np.array of shape (n,) assigning each of the points in X to one of M partitions
    '''
    M = len(models)
    n = len(y)
    Z = []
    for i in range(n):
        
        min_predict_y = 9999999
        target_m = 0
        for m in range(M):
            pred_y = models[m].predict([X[i]])
            
            if (pred_y - y[i])**2 < (min_predict_y - y[i])**2:
                min_predict_y = pred_y
                target_m = m
        Z.append(target_m)
    return np.array(Z)

def get_new_models(x, y, z, models):
    M = len(models)
    n = len(y)
    for m in range(M):
        data_x_m = []
        data_y_m = []
        for i in range(n):
            if (Z[i] == m):
                data_x_m.append(x[i])
                data_y_m.append(y[i])
        if (len(data_x_m) > 0):
            models[m] = LinearRegression().fit(data_x_m, data_y_m)
    return models

n = 400

train_losses = []
test_losses = []
for m in range(1, 31):
    Z = np.array([i % m for i in range(n)])
    models = [LinearRegression().fit(x_train, y_train) for i in range(m)]
    models = get_new_models(x_train, y_train, Z, models)
    loss = total_loss(x_train, y_train, Z, models)
    loss_before = loss 
    
    while (loss_before > loss):
        Z = find_partitions(x_train, y_train, models)
        models = get_new_models(x_train, y_train, Z, models)
        loss_before = loss
        loss = total_loss(x_train, y_train, Z, models)
        
    train_losses.append(loss)
    Z_test = find_partitions(x_test, y_test, models)
    test_losses.append(total_loss(x_test, y_test, Z_test, models)[0])
    print("M is equal to: ", m)
    print("loss train is: ", train_losses[-1])
    print("loss test is: ", test_losses[-1])


ax = plt.gca()
ax.plot(range(1,31), train_losses, color='b', label='Train')
ax.plot(range(1,31), test_losses, color='r', label='Test')
ax.legend()
plt.xlabel("M values")
plt.ylabel("train losses")
plt.title('Losses')
plt.show




