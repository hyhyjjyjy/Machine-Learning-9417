import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

A = np.array([[1,0,1,-1],[-1,1,0,2],[0,-1,-2,1]])
b = np.array([[1],[2],[3]])
x = np.array([[1],[1],[1],[1]])
alpha = 0.1

k  = 0
all_xs = []
alphas = []
alphas.append(0.1)
all_xs.append(x)
while 1 == 1:
    diff_x = np.dot(A.T,(np.dot(A, x) - b))
    x = x - alpha * diff_x
    x_norm = np.linalg.norm(diff_x, ord=2, axis=None)
    all_xs.append(x)
    C = np.dot(np.dot(A,A.T),b) -  np.dot(np.dot(np.dot(A,A.T),A), x)
    D = b - np.dot(A,x)
    alpha = np.dot(C.T, D)/np.dot(C.T, C)
    alphas.append(alpha[0][0])
    if (x_norm < 0.001):
        break
    k += 1
    

print(x_norm)
for i in range(5):
    print(f"k={i},  x({i})={all_xs[i].reshape(4)} ")
for i in range(k - 4, k + 1):
    print(f"k={i},  x({i})={all_xs[i].reshape(4)} ")

plt.plot(alphas)
plt.show()

