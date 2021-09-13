# use classifier instead of regressor
# binary classification
# clf1 saving folder: re_clf1_ + n_feature + n_training_samples

import copy
import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
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

    feature_file = '../saved data/rfe2_204116/rfe2_dict_204116'
    n_features = 5270  # [5270, 1907, 1000, 700, 500, 400, 300, 200, 100, 70, 60, 50, 40, 30, 20, 10, 5, 1]

    df_dataset_file = '../saved data/df_dataset'
    data_portion = 1
    n_class = 2

    clf = RandomForestClassifier(n_estimators=200, n_jobs=10, verbose=1)

    save_data = True

    print('Preprocessing data ...')
    features = pick_features(feature_file=feature_file, feature_number=n_features)
    X_train, y_train, X_test, y_test = preprocessing(features=features, df_dataset_file=df_dataset_file,
                                                     data_portion=data_portion, n_class=n_class)
    print('n_features: ' + str(X_train.shape[1]))
    print()
    print('data_portion: ' + str(data_portion))
    print('n_training_samples: ' + str(X_train.shape[0]))
    print('n_testing_samples: ' + str(X_test.shape[0]))
    print()
    print('----- label distribution in training set:')
    print(y_train.value_counts())
    print('----- label distribution in testing set:')
    print(y_test.value_counts())
    t1 = time.perf_counter()
    print('===== finish preparing dataset, time needed: ' + str(t1 - start))
    print()
    print('Training models ...')
    clf.fit(X_train, y_train)
    t2 = time.perf_counter()
    print('===== finish training the classifier, time needed: ' + str(t2 - t1))
    print()

    print('===== predicting on testing set ...')
    # mean accuracy
    score = clf.score(X_test, y_test)
    print('----- prediction score: ')
    print(score)
    print()

    # confusion matirx
    y_ = clf.predict(X_test)
    print('----- label distribution in predicted values:')
    print(pd.Series(y_).value_counts())
    print()
    confusion_matrix_ = confusion_matrix(y_test, y_, labels=['N', 'Y'])
    print('----- confusion matrix: (row-true(N,Y), column-predict(N,Y)')
    print(confusion_matrix_)
    NN = confusion_matrix_[0, 0]
    NY = confusion_matrix_[0, 1]
    YN = confusion_matrix_[1, 0]
    YY = confusion_matrix_[1, 1]
    print(str(NY) + ' peptides should be N but predicted as Y')
    print(str(YN) + ' peptides should be Y but predicted as N')
    print('True N rate: ' + str(NN / (NN + NY)))
    print('True Y rate: ' + str(YY / (YY + YN)))
    print()

    print('===== predicting on training set ...')
    print(clf.score(X_train, y_train))
    y_pred_train = clf.predict(X_train)
    print(confusion_matrix(y_train, y_pred_train, labels=['N', 'Y']))

    if save_data:
        save_data_folder = '../saved data/rf_clf_1_' + str(n_features) + '_' + str(X_train.shape[0])
        if not os.path.exists(save_data_folder):
            os.makedirs(save_data_folder)
        data_dict = {'clf': clf, 'features': features, 'score': score, 'confusion_matrix': confusion_matrix_,
                     'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'y_': y_}
        save_object(data_dict, save_data_folder + '/data_dict_clf1_' + str(n_features) + '_' + str(X_train.shape[0]))

    end = time.perf_counter()
    print('===== finish saving data, total time passed: ' + str(end - start))


if __name__ == '__main__':
    main()
