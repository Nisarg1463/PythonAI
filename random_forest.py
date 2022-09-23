# %% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# %%
data = pd.read_csv('Position_Salaries.csv')
data

# %%
x = data.iloc[:,1:2].values
y = data.iloc[:,2].values

# %%
reg = RandomForestRegressor(n_estimators = 1500, random_state=0)
reg.fit(x,y)

# %%
y_pred = reg.predict(np.array([6]).reshape(1, -1))
y_pred

# %%
x_grid = np.arange(1, 10, 0.01).reshape((-1,1))
plt.scatter(x, y)
plt.plot(x_grid, reg.predict(x_grid))
plt.show()