# %% importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# %% dataset
data = pd.read_csv('Salary_Data.csv')
data
# %% separating data variables
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values

x = x.reshape(-1,1)
y = y.reshape(-1,1)
# %% dataset division
training_x, test_x, training_y, test_y = train_test_split(x, y, test_size=0.3, random_state=0)
len(training_x)

# %% testing

lin = LinearRegression()
lin.fit(training_x, training_y)

# %% prediction

pred_y = lin.predict(test_x)

# %% comparing values

print(test_y[4])
print(pred_y[4])

# %% ploting values

plt.scatter(training_x, training_y, color='red')
plt.plot(training_x, lin.predict(training_x), color='orange')
plt.title('training data')
plt.ylabel('salary')
plt.xlabel('exp')
plt.show()

plt.scatter(test_x, test_y, color='blue')
plt.plot(test_x, pred_y)
plt.title('testing data')
plt.ylabel('salary')
plt.xlabel('exp')
plt.show()

# %% equation verification

# equation :- y = mx + c

print(lin.coef_)
print(lin.intercept_)
print(lin.coef_*10.3 + lin.intercept_)
print(data)
