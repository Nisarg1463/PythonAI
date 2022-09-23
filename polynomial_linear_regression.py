# %% importing libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# %%
data = pd.read_csv('Position_Salaries.csv')
data.head()

# %%

x = data.iloc[:,1:2].values
y = data.iloc[:,2].values

# %% 
linear_reg = LinearRegression()
linear_reg.fit(x, y)

# %% 
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

linear_reg2 = LinearRegression()
linear_reg2.fit(x_poly, y)

# %% 
plt.scatter(x,y, color='green')
plt.plot(x, linear_reg.predict(x), color='orange')
plt.title('linear regression')
plt.ylabel('salary')
plt.xlabel('position')
plt.show()

# %%
plt.scatter(x,y, color='green')
plt.plot(x, linear_reg2.predict(x_poly), color='orange')
plt.title('polynomial regression')
plt.ylabel('salary')
plt.xlabel('position')
plt.show()

# %%
linear_reg2.predict(poly_reg.fit_transform(np.array(6.5).reshape(1,-1)))