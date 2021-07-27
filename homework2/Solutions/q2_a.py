import numpy as np
A = np.array([[1,0,1,-1],[-1,1,0,2],[0,-1,-2,1]])
b = np.array([[1],[2],[3]])
x = np.array([[1],[1],[1],[1]])
alpha = 0.1

k  = 0
all_xs = []
while 1 == 1:
    diff_x = np.dot(A.T,(np.dot(A, x) - b))
    x = x - alpha * diff_x
    x_norm = np.linalg.norm(diff_x, ord=2, axis=None)
    if (x_norm < 0.001):
        break
    all_xs.append(x)
    k += 1

print(x_norm)
for i in range(5):
    print(f"k={i},  x({i})={all_xs[i].reshape(4)} ")
for i in range(k - 5, k):
    print(f"k={i},  x({i})={all_xs[i].reshape(4)} ")



