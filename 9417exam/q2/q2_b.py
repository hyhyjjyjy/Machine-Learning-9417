import numpy as np
import pandas as pd # not really needed, only for preference
import matplotlib.pyplot as plt
def per(n):
    lists = []
    for i in range(1<<n):
        s=bin(i)[2:]
        s='0'*(n-len(s))+s
        lists.append(list(map(int,list(s))))
    return lists
def train_perceptron(X_data, y, size, max_iter=10000):
    
    eta=1
    np.random.seed(1)
    w = np.array([0 for i in range(size + 1)])       
    nmb_iter = 0
    for _ in range(max_iter):        
        X = X_data
        nmb_iter += 1    
        yXw = (y * X) @ w.T   
        mistake_idxs = np.where(yXw <= 0)[0]
        if mistake_idxs.size > 0:
            i = np.random.choice(mistake_idxs)       
            w = w + y[i] * X[i]            
#             print(f"Iteration {nmb_iter}: w = {w}")
        else: # no mistake made
            print(f"Converged after {nmb_iter} iterations")
            return w, nmb_iter
    print("Cannot converge")
    return w,nmb_iter


def get_y(target_x, size):
    strings = per(size)  
    y = [-1 for i in range(len(strings))]
    for i in range(len(strings)):
        if strings[i] in target_x:
            y[i] = 1
    return y


def do_perceptron(size, target_x):
    target_y = get_y(target_x, size)
    target_y = np.array(np.mat(target_y).T)
    strings  = per(size)
    for string in strings:
        string.insert(0, 1)
    strings = np.array(strings)
    train_perceptron(strings, target_y, size, 10000)


target_x = [[0,1,1], [1,0,0], [1,1,0], [1,1,1]]
do_perceptron(3, target_x)

target_x = [[0,1,0], [0,1,1], [1,0,0], [1,1,1]]
do_perceptron(3, target_x)

target_x = [[0,1,0,0], [0,1,0,1], [0,1,1,0], [1,0,0,0], [1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1]]
do_perceptron(4, target_x)

target_x = [[1,0,0,0,0,0,0], [1,0,0,0,0,0,1], [1,0,0,0,1,0,1]]
do_perceptron(7, target_x)