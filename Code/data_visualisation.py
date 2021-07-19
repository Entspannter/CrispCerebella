# file for all the data visualisation code
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import os
from pandas import read_csv
from pandas import DataFrame
import numpy as np
from numpy import mean
from numpy import std
from numpy import array
from numpy import vstack
from numpy import dstack
from numpy import unique
from scipy import stats
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from data_loading import *
from numpy import vstack
from data_loading import *
from IPython.display import display


# In this section we will assign the classes (which correspond the motions 1-6)
# to each row of the 128 data points.
def rows_by_class(our_data):
    # convert the numpy array into a dataframe
    df_data = pd.DataFrame(our_data, columns=["Class"])
    # grouping the rows by class value and count the amount of rows
    data_grouped_size = df_data['Class'].value_counts().sort_index()
    df_grouped = pd.DataFrame(data_grouped_size)
    # restore raw rows
    data_grouped_size = data_grouped_size.values
    # summary with percetage
    for i in range(len(data_grouped_size)):
        class_percent = data_grouped_size[i] / len(df_data) * 100
        print(f'Class={i + 1}', f'total={data_grouped_size[i]}', f'percentage={class_percent}', sep="\t \t \t")
    return df_grouped


# load files and get a summary of the different class values

trainy = datasetloader('train', '../Dataset/UCI HAR Dataset/')

testy = datasetloader('test', '../Dataset/UCI HAR Dataset/')

all_together = vstack((trainy, testy))
print('Train Dataset\n')
trainy_grouped = rows_by_class(trainy)
print('\nTest Dataset\n')
testy_grouped = rows_by_class(testy)
print('\nTest and Train Dataset (All)\n')
both_grouped = rows_by_class(all_together)

#visualize activity in both 'train' and 'test'
all_grouped_dfs = pd.concat([trainy_grouped, testy_grouped], axis=1, join='inner')
all_grouped_dfs.columns = ['Train','Test']
all_grouped_dfs['Activity Labels'] = ['WALKING', 'WALKING UPSTAIRS', 'WALKING DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
all_grouped_dfs.set_index(['Activity Labels'], inplace=True)
display(all_grouped_dfs)
ax = all_grouped_dfs.plot.bar(color = ('#0000e6', '#ccccff'), figsize=(8,6))
plt.xticks(rotation=75)
plt.ylabel("Count", fontsize=12)
plt.xlabel('Activity Labels', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12)

# get the features from the file features.txt
features = list()
with open("../Dataset/UCI HAR Dataset/features.txt") as f:
 features = [line.split()[1] for line in f.readlines()]
print("No of Features: {}".format(len(features)))
print(features)

# get the test data from txt files to pandas dataffame
X_test = pd.read_csv('../Dataset/UCI HAR Dataset/test/X_test.txt', delim_whitespace=True, header=None)
print(X_test.head())
# add subject column to the dataframe
X_test['subject'] = pd.read_csv('../Dataset/UCI HAR Dataset/test/subject_test.txt', header=None, squeeze=True)

# get y labels from the txt file
y_test = pd.read_csv('../Dataset/UCI HAR Dataset/test/y_test.txt', names=['Activity'], squeeze=True)
y_test_labels = y_test.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',\
                       4:'SITTING', 5:'STANDING',6:'LAYING'})
# put all columns in a single dataframe
test_sub = X_test
test_sub['Activity'] = y_test
test_sub['ActivityName'] = y_test_labels
test_sub.sample()

# get the train data from txt files to pandas dataffame
X_train = pd.read_csv('../Dataset/UCI HAR Dataset/train/X_train.txt', delim_whitespace=True, header=None)
# add subject column to the dataframe
X_train['subject'] = pd.read_csv('../Dataset/UCI HAR Dataset/train/subject_train.txt', header=None, squeeze=True)
y_train = pd.read_csv('../Dataset/UCI HAR Dataset/train/y_train.txt', names=['Activity'], squeeze=True)
y_train_labels = y_train.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',\
                       4:'SITTING', 5:'STANDING',6:'LAYING'})
# put all columns in a single dataframe
train_sub = X_train
train_sub['Activity'] = y_train
train_sub['ActivityName'] = y_train_labels
train_sub.sample()

#visualize activity in both 'train' and 'test'
all_grouped_dfs = pd.concat([trainy_grouped, testy_grouped], axis=1, join='inner')
all_grouped_dfs.columns = ['Train','Test']
all_grouped_dfs['Activity Labels'] = ['WALKING', 'WALKING UPSTAIRS', 'WALKING DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
all_grouped_dfs.set_index(['Activity Labels'], inplace=True)
display(all_grouped_dfs)
ax = all_grouped_dfs.plot.bar(color = ('#0033cc', '#ccccff'), figsize=(5,5))
plt.xticks(rotation=75)
plt.ylabel("Count", fontsize=12)
plt.xlabel('Activity Labels', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.figure(figsize=(30,12))

#Visualize activity distribution for each subject
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Dejavu Sans"
plt.figure(figsize=(12,8))
#plt.title("Data provided by Each Subject (Test Data)", fontsize=20)
sns.countplot(x="subject",hue="ActivityName", data = test_sub, palette="light:b")
sns.color_palette("light:b")
plt.ylabel("Count", fontsize=12)
plt.xlabel('Subjects', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

#Visualize activity distribution for each subject
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Dejavu Sans"
plt.figure(figsize=(12,8))
#plt.title("Data provided by Each Subject (Test Data)", fontsize=20)
sns.countplot(x="subject",hue="ActivityName", data = train_sub, palette="light:b")
sns.color_palette("light:b")
plt.ylabel("Count", fontsize=12)
plt.xlabel('Subjects', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# acquiring individual subject-based data
def individual_timeseries(X, y, sub_map, sub_id):
    # get row indexes for the subject id
    sub_ind_elems = [i for i in range(len(sub_map)) if sub_map[i] == sub_id]
    # return the selected samples
    return X[sub_ind_elems, :, :], y[sub_ind_elems]


# convert a series of windows to a 1D list
def to_series(windows):
    series = []
    for window in windows:
        # remove the overlap from the window
        half = int(len(window) / 2) - 1
        for value in window[-half:]:
            series.append(value)
    return series


# plot the data for one subject
def plot_subject(subX, suby):
    plt.figure(figsize=(15, 20))
    # determine the total number of plots
    n, off = subX.shape[2] + 1, 0
    # plot total acc
    dimensions = ["X", "Y", "Z"]
    # plot activities

    for i in range(3):
        plt.subplot(n, 1, off + 1, )
        plt.plot(to_series(subX[:, :, off]), color='#0000e6')
        plt.title('Total Acc ' + dimensions[i], y=0, loc='left')
        plt.xlabel("Time Steps 50 Hz")
        off += 1
    # plot body acc
    for i in range(3):
        plt.subplot(n, 1, off + 1)
        plt.plot(to_series(subX[:, :, off]), color='#0000e6')
        plt.title('Body Acc ' + dimensions[i], y=0, loc='left')
        off += 1
    # plot body gyro
    for i in range(3):
        plt.subplot(n, 1, off + 1)
        plt.plot(to_series(subX[:, :, off]), color='#0000e6')
        plt.title('Body Gyro ' + dimensions[i], y=0, loc='left')
        off += 1
    # plot activities
    plt.subplot(n, 1, n)
    plt.plot(suby, color='#0000e6')
    plt.title('Activity', y=0, loc='left')


# load data
plt.show()
# load mapping of rows to subjects
sub_map = dataloader('/content/CrispCerebella/Dataset/UCI HAR Dataset/train/subject_train.txt')
train_subjects = unique(sub_map)
print("Subjects in the training set:", train_subjects)
# get the data for one subject
sub_id = train_subjects[16]
subX, suby = individual_timeseries(trainX, trainy, sub_map, sub_id)
print(f'Data for subject No. {sub_id}:', subX.shape, suby.shape, " \n  \n \n ")

# plot data for subject
plot_subject(subX, suby)