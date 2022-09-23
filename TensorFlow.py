# %%
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
from IPython.display import clear_output


# %%
dftrain = pd.read_csv('train.csv')
dfeval = pd.read_csv('eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
dftrain['parch']
# %%
cat_data = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
num_data = ['age', 'fare']

feature_column = []

for feature in cat_data:
    vocabulary = dftrain[feature].unique()
    feature_column.append(tf.feature_column.categorical_column_with_vocabulary_list(feature, vocabulary))

for feature in num_data:
    feature_column.append(tf.feature_column.numeric_column(feature, dtype=tf.float32))

# %%
def make_input_function(dfdata, lable,number_of_epochs = 10, batch_size = 32, shuffle = True):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(dfdata), lable))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(number_of_epochs)
        return ds
    return input_function

# %%
train_input_function = make_input_function(dftrain, y_train)
eval_input_function = make_input_function(dfeval, y_eval, number_of_epochs = 1, shuffle=False)

# %%
linear_estimator = tf.estimator.LinearClassifier(feature_column)

linear_estimator.train(train_input_function)

result1 = linear_estimator.evaluate(eval_input_function)
clear_output()
print(result1['accuracy'])
