#  scale down the number of features using RFECV
#  still use dragontemp.tsv (P1_df.pkl)
#  reference:
#       https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

"""
parameters
"""
df_file = "../saved data/P1_df"
need_fitting = False
#  for rfecv
estimator = RandomForestRegressor(n_estimators=200, max_depth=None)
min_features_to_select = 50
step = 50

"""
load in data
"""
#  read in the dataset
with open(df_file, 'rb') as inp:
    df = pickle.load(inp)

print('size of [#pep, #label+feature]: -------')
print(df.shape)
print('distribution of labels: ------')
print(df['score'].value_counts())

label = 'score'
train, test = train_test_split(df)

#  will use training set for cross validation
y_train = train[label]
X_train = train.drop(label, 1)

"""
use the model
"""
if need_fitting:
    rfecv = RFECV(estimator=estimator, step=step, cv=StratifiedKFold(5),
                  scoring='r2',
                  min_features_to_select=min_features_to_select)

    rfecv.fit(X_train, y_train)

    with open("../saved data/P3_rfecv", 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(rfecv, outp, pickle.HIGHEST_PROTOCOL)

else:
    with open("../saved data/P3_rfecv", 'rb') as inp:
        rfecv = pickle.load(inp)

print(rfecv.grid_scores_)
print(len(rfecv.grid_scores_))

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("subset features")
plt.ylabel("Cross validation score (r2_score)")
plt.plot(range(0, len(rfecv.grid_scores_)),
         rfecv.grid_scores_)
plt.show()
