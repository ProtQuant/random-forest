# recursively reduce features,
# the training set will not differs when eliminating

import copy
import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from matplotlib import pyplot as plt
from sklearn.base import clone


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as inp:
        data = pickle.load(inp)
    return data


def read_in_scores(score_file):
    # read in scores from file
    df_s = pd.read_csv(score_file, sep=',')
    print('finish read in scores: ---------------')
    print(df_s)
    return df_s


def preprocess_scores(df_s, df_s_file):
    print('preprocessing scores: ---------------')

    print('drop the first column (the added index column)')
    df_s.drop(df_s.columns[0], axis=1, inplace=True)
    # print(df_s)

    print('set peptide_formula as key, in order to inner join with peptide_features later')
    df_s.set_index(['peptide'], inplace=True)

    print('save df_s to ' + df_s_file)
    save_object(df_s, df_s_file)

    print(df_s)

    return df_s


def read_in_features(feature_file, chunksize=-1):
    # read in features from file
    if chunksize < 0:
        df_f = pd.read_csv(feature_file, sep='\t')
    else:
        print('read in drangon data by chunks, will create a whole dataframe for it ---------------')
        features = pd.read_csv(feature_file, sep='\t', chunksize=chunksize)
        chunkIndex = 1
        df_f = pd.DataFrame()
        for chunk in features:
            df_f = df_f.append(chunk)
            print("have read in " + str(chunkIndex * chunksize) + " lines in dragon.txt")
            chunkIndex += 1

    print('finish read in features: ---------------')
    print(df_f)
    return df_f


def preprocess_features(df_f, df_f_file='../saved data/df_f', need_save=True):
    print('preprocessing features: ---------------')
    print("replace NaN and 'na' with 0")
    df_f.fillna(0, inplace=True)
    df_f.replace('na', 0, inplace=True)

    print("drop column 'No.'")
    df_f.drop(['No.'], axis=1, inplace=True)

    print("delete '_Peptide ' in 'Name' column")
    df_f['NAME'] = df_f.apply(lambda x: x['NAME'][9:], axis=1)

    print('set peptide_formula as key, in order to inner join with peptide_scores later')
    df_f.set_index(['NAME'], inplace=True)

    if need_save:
        print('save df_f to ' + df_f_file)
        save_object(df_f, df_f_file)

    print(df_f)

    return df_f


def merge_score_and_feature(df_s, df_f):
    df_dataset = pd.merge(df_s, df_f, left_index=True, right_index=True)
    print('finish inner join of scores and features: ---------------')
    print(df_dataset)
    return df_dataset


def data_screening(df_dataset, df_dataset_file):
    print('preprocessing the dataset: ---------------')
    # check data type of each column
    # pd.set_option('display.max_rows', 6000)
    # print(df_dataset.dtypes)

    print("cast int into float, or dtype of a column may be object")
    df_dataset = df_dataset.astype('float64')
    # print(df_dataset.dtypes)

    print('delete the columns has only one value')
    df_dataset.drop(df_dataset.columns[df_dataset.std(ddof=0) == 0], axis=1, inplace=True)

    print('delete duplicate columns')
    df_dataset = df_dataset.T.drop_duplicates().T

    print('save df_dataset to ' + df_dataset_file)
    save_object(df_dataset, df_dataset_file)

    print(df_dataset)
    return df_dataset


def generate_dataset(need_init_dataset=False, need_init_score=False, need_init_feature=False, chunk_size=50000,
                     score_file='../input files/score.csv', feature_file="../input files/all_dragon.txt",
                     saveDataFolder="../saved data", df_s_file='../saved data/df_s', df_f_file='../saved data/df_f',
                     df_dataset_file='../saved data/df_dataset'):
    """
    * This function includes
        - reading in and preprocessing the dataframe of scores(df_s) and features(df_f), and
        - deleting useless columns in the dataframe that made from inner joining df_s and df_f to get df_dataset.
    * Purpose of this function
        - to generate a proper dataset(df_dataset) for training and predicting later
    * Structure:
        - If need_init_dataset is set to False, which means df_dataset has been made, and then all that need to be done
    is just load df_dataset from saved data.
        - If need_init_dataset is set to be ture, it will depends on the values of need_init_score and
    need_init_feature to decide whether to preprocess or to just load in the df_s and df_f.

    :parameter:
        - data loading options (boolean) :
            need_init_dataset, need_init_score, need_init_feature, chunk_size(for read in features, read in all if negative)
        - path of input files (string) :
            score_file, feature_file
        - path to store data (string) :
            saveDataFolder, df_s_file, df_f_file, df_dataset_file

    :return:
        df_dataset:
            a dataframe contains the score and chemical features of different peptides
    """
    """
    create needed output folder
    """
    if not os.path.exists(saveDataFolder):
        os.makedirs(saveDataFolder)

    if need_init_dataset:
        """
        create the dataframe of scores 
        """
        if need_init_score:
            df_s = read_in_scores(score_file)
            df_s = preprocess_scores(df_s, df_s_file)
        else:
            df_s = load_object(df_s_file)
            print('finish load scores: ---------------')
            print(df_s)

        """
        create the dataframe of features
        """
        if need_init_feature:
            df_f = read_in_features(feature_file, chunk_size)  # still cannot be read in once for all on the server
            df_f = preprocess_features(df_f, df_f_file)
        else:
            df_f = load_object(df_f_file)
            print('finish load features: ---------------')
            print(df_f)

        """
        create dataset by join and screen
        """
        df_dataset = merge_score_and_feature(df_s, df_f)
        df_dataset = data_screening(df_dataset, df_dataset_file)
    else:
        df_dataset = load_object(df_dataset_file)
        print('finish load the dataset: ---------------')
        print('shape: ' + str(df_dataset.shape))
        # print(df_dataset)

    # print('size of [#pep, #label+feature]: '+str(df_dataset.shape))
    print('distribution of labels: ------')
    print(df_dataset['score'].value_counts())

    return df_dataset


def draw_score_against_length(df_dataset, score_length_pic='../saved data/score_length_pic', score_label='score'):
    """
    Will draw a picture of peptides' length against their score and save it
    :param df_dataset: dataframe, should have the peptides as index and a column storing score
    :param length_score_pic: string, path to save the picture
    :param score_label: string, the name of the score column
    :return: df_ls, a dataframe of peptides' score and length, sorted by length
    """
    # extract the score
    df_ls = pd.DataFrame(df_dataset, columns=[score_label])

    # record the length of each peptide (the index)
    df_ls['length'] = df_ls.apply(lambda x: len(str(x.name)), axis=1)

    # sort by length
    df_ls.sort_values(by='length', inplace=True)

    # print(df_ls)
    plt.figure()
    plt.xlabel("length")
    plt.ylabel("score")
    plt.plot(df_ls['length'], df_ls['score'], '.')
    plt.savefig(score_length_pic)

    print("finish drawing scores against length ---------------")
    return df_ls


def split_dateset(df_dataset, test_size=0.25, label='score'):
    """
    :param df_dataset: Dataframe, index=peptides, columns=score+fetures
    :param test_size: float, (0,1)
    :param label: string, ['score'] column name
    :return: X_train, y_train, X_test, y_test
    """
    train, test = train_test_split(df_dataset, test_size=test_size)
    y_train = train[label]
    X_train = train.drop(label, 1)

    # sort test set by score
    test = test.sort_values(by=[label])
    y_test = test[label]
    X_test = test.drop(label, 1)

    return X_train, y_train, X_test, y_test


def recursive_feature_elimination(X_train, y_train, X_test, y_test, estimator, rfe_dict,
                                  origin_n_features, target_n_features,
                                  save_rfe_dict=True, save_data_folder='../saved data',
                                  draw_prediction_pic=True, save_pic_folder='../saved data/rfe prediction/'):
    """
    Altered from sklean rfe.
    Will reduce features from origin_n_features to target_n_features
    Use rfe_dict[origin_n_features]['support'], rfe_dict[origin_n_features]['ranking'] to
    - calculate rfe_dict[origin_n_features]['score'] (r2_score <= 1)
    - generate rfe_dict[target_n_features]['support'], rfe_dict[target_n_features]['ranking']

    :param
    estimator: random forest regressor
    origin_n_features: int, an existing key in rfe_dict
    target_n_features: int, smaller than origin_n_features

    :return:
    rfe_dict: dictionary, storing masks for features (support), feature ranking (most important : 1), r2_score on test
              set (default to 9999, should be no larger than 1)
                n_features: {'support': [], 'ranking': [], 'score': 9999}
              How to use:
                indices of needed features:
                    support = rfe_dict[target_n_features]['support']
                    features = np.arange(n_all_features)[support]
                Select these features in test set:
                    X_test.iloc[:, features], y_test
    """
    print()
    n_features = X_train.shape[1]
    n_training_samples = X_train.shape[0]
    support = copy.deepcopy(rfe_dict[origin_n_features]['support'])
    ranking = copy.deepcopy(rfe_dict[origin_n_features]['ranking'])
    estimator = clone(estimator)

    features = np.arange(n_features)[support]

    print('Fitting the estimator, n_features = ' + str(origin_n_features))
    print('Preparing reducing to ' + str(target_n_features))
    estimator.fit(X_train.iloc[:, features], y_train)
    print('predicting ---')
    y_ = estimator.predict(X_test.iloc[:, features])
    score = r2_score(y_test, y_)
    rfe_dict[origin_n_features]['score'] = score
    print('r2_score: ' + str(score))

    importances = estimator.feature_importances_
    ranks = np.argsort(importances)  # sort the indices
    threshold = origin_n_features - target_n_features
    support[features[ranks][:threshold]] = False  # set the unneeded features to false
    ranking[np.logical_not(support)] += 1

    rfe_dict[target_n_features] = {'support': support, 'ranking': ranking, 'score': 9999}
    print([sum(rfe_dict[k]['support']) for k in rfe_dict])
    if save_rfe_dict:
        save_object(rfe_dict, save_data_folder + '/rfe2_dict_' + str(n_training_samples))
        print('have saved rfe2_dict_' + str(n_training_samples))

    if draw_prediction_pic:
        if not os.path.exists(save_pic_folder):
            os.makedirs(save_pic_folder)
        plt.figure()
        plt.title('n_features = ' + str(origin_n_features))
        plt.xlabel('peptides')
        plt.ylabel('score')
        x = range(len(y_test))
        plt.scatter(x, y_)
        plt.scatter(x, y_test)
        plt.savefig(save_pic_folder + '/' + str(origin_n_features))
        print('finish drawing prediction pictures')

    return rfe_dict


def main():
    """ """
    start = time.perf_counter()
    print()

    """
    parameters
    """
    #  data loading options
    need_init_dataset = False
    need_init_score = False
    need_init_feature = False
    chunk_size = 50000

    need_score_length_pic = False
    save_dataset_dict = True

    #  input files
    score_file = '../input files/score.csv'
    feature_file = "../input files/all_dragon.txt"

    # save path
    saveDataFolder = "../saved data"
    df_s_file = '../saved data/df_s'  # df_s is a dataframe transformed from score.csv
    df_f_file = '../saved data/df_f'  # df_f is a dataframe transformed from all_dragon.txt
    df_dataset_file = '../saved data/df_dataset'

    # training
    data_portion = 0.01  # (0,1], set to 1 if using all the peptides
    n = [1907, 1000, 700, 500, 400, 300, 200, 100, 70, 60, 50, 40, 30, 20, 10, 5, 1, 0]  # n_features
    test_size = 0.25
    label = 'score'
    estimator = RandomForestRegressor(n_estimators=200, max_depth=None, verbose=1, n_jobs=10)
    save_rfe_dict = True
    draw_prediction_pic = True
    save_data_folder = '../saved data'
    save_pic_folder = ''

    """
    create dataset
    """
    df_dataset = generate_dataset(need_init_dataset=need_init_dataset, need_init_feature=need_init_feature,
                                  need_init_score=need_init_score, chunk_size=chunk_size, score_file=score_file,
                                  feature_file=feature_file, saveDataFolder=saveDataFolder, df_s_file=df_s_file,
                                  df_f_file=df_f_file, df_dataset_file=df_dataset_file)
    if need_score_length_pic:
        draw_score_against_length(df_dataset)

    if data_portion != 1:
        drop, df_dataset = train_test_split(df_dataset, test_size=data_portion)

    X_train, y_train, X_test, y_test = split_dateset(df_dataset, test_size=test_size, label=label)
    n_training_samples = X_train.shape[0]
    print()
    print('For further processing =================')
    print('n_samples in total: ' + str(df_dataset.shape[0]))
    print('n_samples for training: ' + str(X_train.shape[0]))
    print('n_samples for testing : ' + str(X_train.shape[0]))

    save_data_folder = save_data_folder + '/rfe2_' + str(n_training_samples)
    if not os.path.exists(save_data_folder):
        os.makedirs(save_data_folder)
    save_pic_folder = save_data_folder + '/rfe2_prediction_' + str(n_training_samples)

    if save_dataset_dict:
        dataset_dict = {'n': n, 'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
        save_object(dataset_dict, save_data_folder + '/dataset_dict')

    data_time = time.perf_counter()
    print('finish preparing dataset info, time: ' + str(data_time - start))

    """
    rfe
    """
    print()
    rfe_dict = {}
    n_features = df_dataset.shape[1] - 1
    print('original number of features: ' + str(n_features))
    support = np.ones(n_features, dtype=bool)
    ranking = np.ones(n_features, dtype=int)
    rfe_dict[n_features] = {'support': support, 'ranking': ranking, 'score': 9999}  # score <= 1

    # origin_n_features = n_features  # 1907
    # target_n_features = 1000

    o_n = n[:-1]
    t_n = n[1:]
    for origin_n_features, target_n_features in zip(o_n, t_n):
        rfe_dict = recursive_feature_elimination(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 estimator=estimator, rfe_dict=rfe_dict,
                                                 origin_n_features=origin_n_features,
                                                 target_n_features=target_n_features,
                                                 save_rfe_dict=save_rfe_dict, save_data_folder=save_data_folder,
                                                 draw_prediction_pic=draw_prediction_pic,
                                                 save_pic_folder=save_pic_folder)


if __name__ == '__main__':
    main()
