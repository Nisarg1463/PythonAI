# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# %%
data = pd.read_csv('Social_Network_ads.csv')
data

# %%
x = data.iloc[:,[2,3]].values
y = data.iloc[:,[4]].values
print(x,y)

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.25)

#%% getting data in valid range (-2,2)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# %%
classifier_lr = LogisticRegression(random_state=0)
classifier_lr.fit(x_train, y_train)
# %%
y_pred = classifier_lr.predict(x_test)

# %% checking accuracy
cm = confusion_matrix(y_test, y_pred)
cm # diagonal is correct matches and other two are incorrect so correct = add diagonal values and incorrect = add other two values

(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100
y_train
# %%
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1, stop = x_set[:,0].max()+1, step = 0.01), np.arange(start = x_set[:,1].min()-1, stop = x_set[:,1].max()+1, step = 0.01))
plt.contourf(x1, x2, classifier_lr.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),alpha= 0.50, cmap=ListedColormap(('red','green'))) # it will divide data logistic regression and ravel function will fing midpoint
y_set = y_set.flatten()
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.xlabel('Age')
plt.ylabel('salary')
plt.legend()
plt.show()
# %%
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1, stop = x_set[:,0].max()+1, step = 0.01), np.arange(start = x_set[:,1].min()-1, stop = x_set[:,1].max()+1, step = 0.01))
plt.contourf(x1, x2, classifier_lr.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),alpha= 0.50, cmap=ListedColormap(('red','green'))) # it will divide data logistic regression and ravel function will fing midpoint
y_set = y_set.flatten()
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.xlabel('Age')
plt.ylabel('salary')
plt.legend()
plt.show()