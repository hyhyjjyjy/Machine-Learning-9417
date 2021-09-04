import numpy as np
a_result = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
b_result = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0]
c_result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]

def print_false(betas, result):
    False_Positives = 0
    False_Negatives = 0
    for i in range(len(betas)):
        if (betas[i] == 0 and result[i] != 0):
            False_Negatives += 1
        if (betas[i] != 0 and result[i] == 0):
            False_Positives += 1
    print(False_Positives," ", False_Positives)


# NP 
np.random.seed(125)
p = 30
k = 8
betas = np.random.random(p) + 1
new_betas = betas
new_betas[np.random.choice(np.arange(p), p-k, replace=False)] = 0.0
print_false(new_betas, a_result)

k = 9
betas = np.random.random(p) + 1
new_betas = betas
new_betas[np.random.choice(np.arange(p), p-k, replace=False)] = 0.0
print_false(new_betas, b_result)

k = 3
betas = np.random.random(p) + 1
new_betas = betas
new_betas[np.random.choice(np.arange(p), p-k, replace=False)] = 0.0
print_false(new_betas, c_result)
