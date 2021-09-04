import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# load the diabetes dataset
df=pd.read_csv('data.csv')
target_name="Y"
target=df[target_name]

stand = np.std(df)
newdf = (df - df.mean())/np.std(df)
for num in range(1, 9):
    sum = 0
    index = "X" + str(num)
    for i in newdf[index]:
        sum += i**2
    print(index, sum)