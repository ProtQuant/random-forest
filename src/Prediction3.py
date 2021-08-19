## 要排序
## all_dragon is too big
# os.chdir('/home/sgyzh180/ProtQuant/random forest/src/')

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def save_object(obj, filename):  # filename: ".pkl"
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
    if chunksize < 0:
        df_f = pd.read_csv(feature_file, sep='\t')
    else:
        print('read in drangon data by chunks, will create a whole dataframe for it ---------------')
        features = pd.read_csv(feature_file, sep='\t', chunksize=chunk_size)
        chunkIndex = 1
        df_f = pd.DataFrame()
        for chunk in features:
            df_f = df_f.append(chunk)
            print("have read in " + str(chunkIndex * chunk_size) + " lines in dragon.txt")
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
    """

    :param df_dataset:
    :param df_dataset_file:
    :return:
    """
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


def generate_dataset(need_init_dataset = False, need_init_score = False, need_init_feature = False, chunk_size = 50000,
                     score_file = '../input files/score.csv', feature_file = "../input files/all_dragon.txt",
                     df_s_file = '../saved data/df_s', df_f_file = '../saved data/df_f',
                     df_dataset_file = '../saved data/df_dataset'):
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
            need_init_dataset, need_init_score, need_init_feature, chunk_size(for read in features)
        - path of input files (string) :
            score_file, feature_file
        - path to store data (string) :
             df_s_file, df_f_file, df_dataset_file

    :return:
        df_dataset:
            a dataframe contains the score and chemical features of different peptides
    """

"""
parameters
"""
#  data loading options
need_init_dataset = True
need_init_score = True
need_init_feature = True
chunk_size = 50000

need_init_dataset = False
need_init_score = False
need_init_feature = False

#  input files
# score_file = '../../score-peptides/output/score.csv'
score_file = '../input files/score.csv'
# feature_file = "D:/all_dragon.txt"
feature_file = "../input files/all_dragon.txt"

# default values
df_s_file = '../saved data/df_s'  # df_s is a dataframe transformed from score.csv
df_f_file = '../saved data/df_f'  # df_f is a dataframe transformed from all_dragon.txt
df_dataset_file = '../saved data/df_dataset'

"""
create needed output folder
"""
saveDataFolder = "../saved data"
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
    print('finish load features: ---------------')
    print(df_dataset)



# """
# merge score and feature when read in the features
# """
# chunk_size = 50000
# features = pd.read_csv(feature_file, sep='\t', chunksize=chunk_size)
# chunkIndex = 1
# data_set = pd.DataFrame()  # scores and features
# for df_f in features:
#     # pre-process features
#     df_f = preprocess_features(df_f)
#     # inner join scores and features
#     df = pd.merge(df_s, df_f, left_index=True, right_index=True)
#     # concatenate data_set
#     data_set = data_set.append(df)
#     print("have processed " + str(chunkIndex * chunk_size) + " lines in dragon.txt")
#     chunkIndex += 1
#     print(data_set)
#
# save_object(data_set, '../saved data/data_set.pkl')
#
# print('size of [#pep, #label+feature]: -------')
# print(data_set.shape)
# # find out the distribution of scores
# print('distribution of labels: ------')
# print(data_set['score'].value_counts())
