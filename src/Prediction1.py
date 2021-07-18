import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

"""
read in the chemical features
"""
feature_file = "../dragontemp.tsv"
df_f = pd.read_csv(feature_file, sep='\t')  # chunksize=1000000 or iterator if the file is too large
# print(df)

# replace tne 'na' with 0. They are string, not really NaN
df_f.replace('na', 0, inplace=True)
# print(df)

# drop column 'No.'
df_f.drop(['No.'], axis=1, inplace=True)
# print(df)

# delete '_Peptide ' in 'Name' column
df_f['NAME'] = df_f.apply(lambda x: x['NAME'][9:], axis=1)
# print(df_f)

"""
read in the scores
"""
score_file = '../../score-peptides/Testcases4/T2/output/score.csv'
df_s = pd.read_csv(score_file, sep=',')
# print(df_s)

# drop the first column
df_s.drop(df_s.columns[0], axis=1, inplace=True)
# print(df_s)

"""
inner join features and scores using peptides' formulas
"""
df_f.set_index(['NAME'], inplace=True)  # should have no duplicate peptides
# print(df_f)
df_s.set_index(['peptide'], inplace=True)
# print(df_s)
df = pd.merge(df_s, df_f, left_index=True, right_index=True) 
print(df)
# find out the distribution of scores
print(df['score'].value_counts())
"""
data screening is needed here
"""

"""
build train and test data set
"""
label = 'score'
y = df[label]
X = df.drop(label, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y)
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

"""
Random forest regression
"""
# num_trees = 200
# Rf = RandomForestRegressor(n_estimators=num_trees, max_depth=None)
#
# Rf.fit(X_train, Y_train)
# y_ = Rf.predict(X_test)
#
# print(y_, Y_test)
# print(r2_score(Y_test, y_))
#
# x = range(len(Y_test))
# plt.scatter(x, Y_test )
# plt.scatter(x, y_)
# plt.show()
