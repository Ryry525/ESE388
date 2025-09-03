import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

array = np.random.randint(1, 100, size=(100, 1))

print(array)

pandasSeries = pd.DataFrame(array, columns=['value'])

print(pandasSeries)

print("---" * 10)
array2 = np.random.randint(1, 100, size=(100, 3))
print("Printing the second array:\n" ,array2)
pandasDataFrame = pd.DataFrame(array2, columns=['value1', 'value2', 'value3'])
print(pandasDataFrame)
sns_plot = sns.pairplot(pandasDataFrame)
plt.show()  