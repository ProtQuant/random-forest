# use classifier instead of regressor
# binary classification

# clf1 overfit too much
# us cross validation to adjust n_estimators [0, 250]
# type -- 0/1, score--recall=TP/(TP+FN)

import copy
import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, ShuffleSplit
from matplotlib import pyplot as plt
from sklearn.base import clone

n_all_features = 1907


def save_object(obj, filename):  # filename: ".pkl"
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as inp:
        data = pickle.load(inp)
    return data


def split_dateset(df_dataset, test_size=0.25, label='score'):
    """
    :param df_dataset: peptides, score+fetures
    :param label: ['score'] volumn name
    :return: X_train, y_train, X_test, y_test
    """

    train, test = train_test_split(df_dataset, test_size=test_size)
    y_train = train[label]
    X_train = train.drop(columns=label, axis=1)

    test = test.sort_values(by=[label])
    y_test = test[label]
    X_test = test.drop(columns=label, axis=1)

    return X_train, y_train, X_test, y_test


def pick_features(feature_file='../saved data/rfe2_204116/rfe2_dict_204116', feature_number=50):
    """
    For now, pick feature_number from [5270, 1907, 1000, 700, 500, 400, 300, 200, 100, 70, 60, 50, 40, 30, 20, 10, 5, 1]
    """
    if feature_number > 1907:
        features = np.arange(5270)
    else:
        rfe_dict = load_object(feature_file)
        support = rfe_dict[feature_number]['support']
        features = np.arange(n_all_features)[support]
    return features


def preprocessing(df_dataset_file='../saved data/df_dataset', df_dataset_file_all='../saved data/df_dataset_all',
                  data_portion=0.5, n_class=2,
                  test_size=0.25, label='score', features=np.arange(n_all_features)):
    """"""
    if len(features) > n_all_features:
        df_dataset_file = df_dataset_file_all
    df_dataset = load_object(df_dataset_file)
    print('----- score distribution in the original dataset:')
    print(df_dataset['score'].value_counts())
    print()

    if data_portion != 1:
        drop, df_dataset = train_test_split(df_dataset, test_size=data_portion)

    if n_class == 2:
        print('Will conduct binary classification')
        print('Peptides scored as 0 are classified as N, others as Y')
        print()
        col_names = df_dataset.columns.tolist()
        col_names.insert(0, 'type')
        df_dataset = df_dataset.reindex(columns=col_names)

        def define_type(x):
            s = x[label]
            if s > 0:
                type = 'Y'
            else:
                type = 'N'
            return type

        df_dataset['type'] = df_dataset.apply(lambda x: define_type(x), axis=1)
        df_dataset = df_dataset.drop(columns=label, axis=1)
        label = 'type'

    X_train, y_train, X_test, y_test = split_dateset(df_dataset, test_size=test_size, label=label)

    X_train = X_train.iloc[:, features]
    X_test = X_test.iloc[:, features]

    return X_train, y_train, X_test, y_test


def main():
    start = time.perf_counter()
    print()
    """
    setting parameters
    """
    feature_file = '../saved data/rfe2_204116/rfe2_dict_204116'
    n_features = 50  # [1907, 1000, 700, 500, 400, 300, 200, 100, 70, 60, 50, 40, 30, 20, 10, 5, 1]

    df_dataset_file = '../saved data/df_dataset'
    data_portion = 0.01

    #  choose needed features
    features = pick_features(feature_file=feature_file, feature_number=n_features)

    #  use a smaller dataset
    df_dataset = load_object(df_dataset_file)
    if data_portion != 1:
        drop, df_dataset = train_test_split(df_dataset, test_size=data_portion)

    #  classify peptides into two types according to scores (0 - 0, 1 - others)
    col_names = df_dataset.columns.tolist()
    col_names.insert(0, 'type')
    df_dataset = df_dataset.reindex(columns=col_names)

    def define_type(x):
        s = x['score']
        if s > 0:
            type = 1
        else:
            type = 0
        return type

    df_dataset['type'] = df_dataset.apply(lambda x: define_type(x), axis=1)
    df_dataset = df_dataset.drop(columns='score', axis=1)

    #  create dataset for cross validation
    y = df_dataset['type']
    X = df_dataset.drop(columns='type', axis=1).iloc[:, features]
    print(y.value_counts())
    print(X.shape)

    #  cross validation
    score_lt = []
    for i in range(0, 250, 10):
        n = i + 1
        clf = RandomForestClassifier(n_estimators=n, n_jobs=10) # random_state=
        s = cross_val_score(clf, X, y, cv=5, scoring='recall')
        print(s)
        score = s.mean()
        score_lt.append(score)
    score_max = max(score_lt)
    print(score_lt)
    print('maximum score:{}'.format(score_max),
          'n_estimators:{}'.format(score_lt.index(score_max) * 10 + 1))

    # plot
    x = np.arange(1, 251, 10)
    plt.subplot(111)
    plt.plot(x, score_lt, 'r-')

    base = '../saved data/' + os.path.basename(__file__).split('.')[0]
    if not os.path.exists(base):
        os.makedirs(base)

    plt.savefig(base+'/cross_val_recall_'+str(X.shape[0]))


if __name__ == '__main__':
    main()
