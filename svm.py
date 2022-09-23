# %% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# %%
data = pd.read_csv('Social_Network_Ads.csv')

# %%
x = data.iloc[:, [2,3]].values
y = data.iloc[:, 4].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
# %%
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
# %%
_svc = SVC(kernel='linear', random_state=0)
_svc.fit(x_train, y_train)

# %%
y_pred = _svc.predict(x_test)

# %%
cm = confusion_matrix(y_test, y_pred)
cm