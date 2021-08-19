## 要排序
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

"""
parameters
"""
need_preprocessing = True
need_fitting = False
feature_file = "../dragontemp.tsv"
score_file = '../../score-peptides/output/score.csv'


if need_preprocessing:
    """
    read in the chemical features
    """
    # feature_file = "../dragontemp.tsv"
    # feature_file = "D:/all_dragon.txt"
    df_f = pd.read_csv(feature_file, sep='\t')  # chunksize=1000000 or iterator if the file is too large
    # print(df)
    print(df_f.shape)

    # replace tne 'na' with 0. They are string, not really NaN
    df_f.fillna(0, inplace=True)
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
    # score_file = '../../score-peptides/output/60_score.csv'
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
    print('size of [#pep, #label+feature]: -------')
    print(df.shape)
    # find out the distribution of scores
    print('distribution of labels: ------')
    print(df['score'].value_counts())

    """
    data screening 
    """
    # check data type of each column
    # pd.set_option('display.max_rows', 6000)
    # print(df.dtypes)

    # may remove columns whose dtype is object (a mixture of int64 and float64)
    # df = df._get_numeric_data()
    # print(df.shape)

    # cast int into float, or dtype of a column may be object
    df = df.astype('float64')
    # print(df.dtypes)

    #  delete the columns has only one value, can be done earlier when read in features file
    # df = df.loc[:, (df != df.iloc[0]).any()]
    # print(df.std(ddof=0) == 0)
    df.drop(df.columns[df.std(ddof=0) == 0], axis=1, inplace=True)
    print('size after drop columns having only one value: ------')
    print(df.shape)  # 5271 --> 2039

    # delete duplicate columns
    df = df = df.T.drop_duplicates().T
    print('size after drop duplicate columns: ------')
    print(df.shape)

    # save df
    with open("../saved data/P1_df", 'wb') as outp:
        pickle.dump(df, outp, pickle.HIGHEST_PROTOCOL)

'''

"""
build train and test data set
"""
if not need_preprocessing:
    with open("../saved data/P1_df", 'rb') as inp:
        df = pickle.load(inp)
    # print(df.shape)
label = 'score'
train, test = train_test_split(df)

y_train = train[label]
X_train = train.drop(label, 1)

test = test.sort_values(by=[label])
y_test = test[label]
X_test = test.drop(label, 1)

# print("label for train and test: ----------")
# print(y_train)
# print(y_test)

# y = df[label]
# X = df.drop(label, 1)
# X_train, X_test, y_train, y_test = train_test_split(X, y)

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

"""
Random forest regression
"""
if need_fitting:
    num_trees = 200
    Rf = RandomForestRegressor(n_estimators=num_trees, max_depth=None)

    Rf.fit(X_train, y_train)
    y_ = Rf.predict(X_test)

    with open("../saved data/P1_Rf", 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(Rf, outp, pickle.HIGHEST_PROTOCOL)

if not need_fitting:
    with open("../saved data/P1_Rf", 'rb') as inp:
        Rf = pickle.load(inp)

# importance of features
features = df.columns[1:]
importances = Rf.feature_importances_
# print(importances.shape)
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, features[indices[f]], importances[indices[f]]))

# print(y_, y_test)
print(r2_score(y_test, y_))

x = range(len(y_test))
plt.scatter(x, y_test )
plt.scatter(x, y_)
plt.show()

'''

