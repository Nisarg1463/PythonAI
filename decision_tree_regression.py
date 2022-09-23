# %% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# %%
data = pd.read_csv('Position_Salaries.csv')

# %%
x = data.iloc[:,1:2]
y = data.iloc[:,2]
# %%
reg = DecisionTreeRegressor(random_state=0)
reg.fit(x, y)

# %%
y_pred = reg.predict(np.array([6]).reshape(1,-1))

# %%
x_grid = np.arange(1,10, 0.01).reshape((-1,1))
plt.scatter(x, y)
plt.plot(x_grid, reg.predict(x_grid))
plt.title('Decision tree regression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()
# %%
