# %% importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# %% loading data
data = pd.read_csv('50_Startups.csv')
data

# %% variables selection
x = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values

le = LabelEncoder()
x[:,3] = le.fit_transform(x[:,3])
ct = ColumnTransformer([('City', OneHotEncoder(), [3])], remainder='passthrough')
x = ct.fit_transform(x)


# %% splitting dataset
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state=0)
train_x

# %% training
mlr = LinearRegression()
mlr.fit(train_x, train_y)

# %% prediction
pred_y = mlr.predict(test_x)

# %% checking prediction
print(pred_y-test_y)

# %% backward elemination
data = pd.read_csv('50_Startups.csv')

x = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values

le = LabelEncoder()
x[:,3] = le.fit_transform(x[:,3])
ct = ColumnTransformer([('City', OneHotEncoder(), [3])], remainder='passthrough')
x = ct.fit_transform(x)

x = x[:,1:]
# %% splitting dataset
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state=0)
train_x

# %% training
mlr = LinearRegression()
mlr.fit(train_x, train_y)

# %% prediction
pred_y = mlr.predict(test_x)

# %% checking prediction
print(pred_y-test_y)

# %% 
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)

# %%
x_opt = np.array(x[:, [0,1,2,3,4,5]], dtype=float)
reg_OLS = sm.OLS(endog = y, exog=x_opt).fit()
reg_OLS.summary()

# %%
x_opt = np.array(x[:, [0,1,2,3,4]], dtype=float)
reg_OLS = sm.OLS(endog = y, exog=x_opt).fit()
reg_OLS.summary()
# %%
x_opt = np.array(x[:, [0,3,4,5]], dtype=float)
reg_OLS = sm.OLS(endog = y, exog=x_opt).fit()
reg_OLS.summary()

# %%
x_opt = np.array(x[:, [0,3,5]], dtype=float)
reg_OLS = sm.OLS(endog = y, exog=x_opt).fit()
reg_OLS.summary()

# %%
x_opt = np.array(x[:, [0,3]], dtype=float)
reg_OLS = sm.OLS(endog = y, exog=x_opt).fit()
reg_OLS.summary()

# %%
train_x, test_x, train_y, test_y = train_test_split(x_opt, y)

# %%
mlr = LinearRegression()
mlr.fit(train_x, train_y)

pred_y = mlr.predict(test_x)
pred_y-test_y
