# %% importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# %%
data = pd.read_csv('Social_Network_Ads.csv')
# %%
x = data.iloc[:, [2,3]].values
y = data.iloc[:, [4]].values

# %%
train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=0, test_size = 0.25)

# %% 
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
test_x = ss.fit_transform(test_x)
# %%
cls = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
cls.fit(train_x, train_y)

# %%
y_pred = cls.predict(test_x)
# %%
cm = confusion_matrix(test_y, y_pred)
cm