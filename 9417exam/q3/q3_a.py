import numpy as np
import pandas as pd # not really needed, only for preference
import matplotlib.pyplot as plt


def plot_perceptron(ax, data_X, data_y, w):
#     print(np.where(y==1)[0])
    X = pd.DataFrame(data_X)
    y = pd.DataFrame(data_y)
    pos_points = X.iloc[np.where(y==1)[0]]
    neg_points = X.iloc[np.where(y==-1)[0]]
    ax.scatter(pos_points[1], pos_points[2], color='blue')
    ax.scatter(neg_points[1], neg_points[2], color='red')
    xx = np.linspace(-6,6)
    yy = -w[0]/w[2] - w[1]/w[2] * xx
    ax.plot(xx, yy, color='orange')
    ratio = (w[2]/w[1] + w[1]/w[2])
    xpt = (-1*w[0] / w[2]) * 1/ratio
    ypt = (-1*w[0] / w[1]) * 1/ratio
    ax.arrow(xpt, ypt, w[1], w[2], head_width=0.2, color='orange')
    ax.axis('equal')

def train_perceptron(X_data, y, max_iter=100):
    
    eta=1
    np.random.seed(1)
    w = np.array([0,0,0])       
    nmb_iter = 0
    for _ in range(max_iter):               # termination condition (avoid running forever)
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

data_x = pd.read_csv('Q3X.csv')
data_y = pd.read_csv('Q3y.csv')
print(data_y.shape)
print(data_x.shape)
data_x = np.array(data_x)
data_y = np.array(data_y)


w,nmb_iter = train_perceptron(data_x, data_y, 100)
fig, ax = plt.subplots()
plot_perceptron(ax, data_x, data_y, w)
ax.set_title(f"w={w}, iterations={nmb_iter}")
plt.show()