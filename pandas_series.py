# %% importing important libraries
import pandas as pd
import numpy as np

# %% variables
string_lst = ['a','b', 'c','d','e','f','g','h','i','j', 'a']
int_lst = np.random.randint(0,12, 11)
bool_lst = [True, False, False, True, True, False, True]
float_lst = np.random.rand(10) * np.random.randint(1, 11, 10)

data = pd.Series(string_lst)
float_data = pd.Series(float_lst)

# %% series objects
pd.Series(int_lst)
pd.Series(bool_lst)

# %% attributes
data.index
data.values
data.shape
data.unique

# %% methods
float_data.sum()
float_data.product()
float_data.mean()

pd.Series(string_lst, int_lst)
pd.Series(int_lst)
# %% Sorting
float_data.sort_values()
float_data.sort_index()

# %% varibles
data = pd.read_csv('myFile0.csv', usecols=['firstname'], squeeze=True)

# %% usage
data.head(10)
data.tail(10)

# %% sorting methods
data.sort_values(ascending=True, inplace=True)
data.sort_index(ascending=True, inplace=True)
data.head()

# %% finding values using indexing
data[10]
data[:21]
data[-31:]
data[10:21:2]
data[[100, 200, 300]]

# %% using different indexing
data = pd.read_csv('myFile0.csv', index_col='firstname')
data.loc['Nickie']

# %% mathematical functions
num_data = pd.Series(np.random.randint(0, 100, 20))
num_data.sum()
num_data.mean()
num_data.median()
num_data.std()
num_data.count()
num_data.min()
num_data.max()
num_data.idxmax()
num_data.idxmin()

# %% more functions
series_data = pd.read_csv('myFile0.csv', usecols=['firstname'], squeeze=True)
series_data.value_counts()

num_data.apply(lambda value: value % 2 == 0)
