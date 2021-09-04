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


def find_mistake_values(w, X, y, r, III):
    sums = []
    target_i = 0
    for i in range(81):
        one_sum = y[i] * np.matmul(w, X[i]) + III[i] * r
        sums.append(one_sum)
    sums = np.array(sums)
    mistake_idxs = np.where(sums <= 0)[0]
    return mistake_idxs

def train_perceptron(X_data, y, max_iter=100, r=2):
    eta=1
    np.random.seed(1)
    III = np.array([0 for i in range(81)])
    w = np.array([0,0,0])
    nmb_iter = 0
    for _ in range(max_iter):               # termination condition (avoid running forever)
        X = X_data
        nmb_iter += 1           
        mistake_idxs = find_mistake_values(w, X, y, r, III)
        if mistake_idxs.size > 0:
            i = np.random.choice(mistake_idxs)       
            w = w + y[i] * X[i]
            III[i] = 1
#             print(f"Iteration {nmb_iter}: alpha = {alpha}")

        else: # no mistake made
            print(f"Converged after {nmb_iter} iterations")
            return w, nmb_iter
    print("Cannot converge")
    return w,nmb_iter

data_x = pd.read_csv('Q3X.csv')
data_y = pd.read_csv('Q3y.csv')
print(data_y.shape)
print(data_x.shape)
data_x = data_x.values
data_y = data_y.values
w, nmb_iter = train_perceptron(data_x, data_y, 100, 2)
n = 81


fig, ax = plt.subplots()
print(w)
plot_perceptron(ax, data_x, data_y, w) # from neural learning lab
ax.set_title(f"w = {w}, iterations={nmb_iter}")
plt.savefig("name.png", dpi=300) # if you want to save your plot as a png
plt.show()