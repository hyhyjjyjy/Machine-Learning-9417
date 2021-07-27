import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df=pd.read_csv('data.csv')
target_name="Y"
target=df[target_name]

sns.pairplot(df)#
plt.show()