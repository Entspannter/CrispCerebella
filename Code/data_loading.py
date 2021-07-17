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
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.utils import to_categorical
import keras_tuner
from keras_tuner import HyperModel
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns


#Build a Dataloader function

def dataloader(d_path):
	dataframe = read_csv(d_path, header=None, delim_whitespace=True)
	return dataframe.values

#data = dataloader('../Dataset/UCI HAR Dataset/train/Inertial Signals/total_acc_y_train.txt')

#print("Shape of Total_Acc_y_train:", data.shape)

# function to load a group of data and create a 3 dimensional tensor out of it
def dataloader_group(all_files, directory=''):
	datalist = list()
 # for each element in the filelist we load each file seperately and append its data to a list
	for elem in all_files:
		data = dataloader(directory + elem)
		datalist.append(data)
	# create a 3 dimensional tensor out of the individual data files
	datalist = dstack(datalist)
	return datalist

# load all acc data into a 3 dimensional tensor 
#total_acc_xyz = dataloader_group(['total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt'], directory='/content/CrispCerebella/Dataset/UCI HAR Dataset/train/Inertial Signals/')
#print("3D Form of Total Acc Data",total_acc_xyz.shape)

# create a train and test group out of the data
def datasetloader(train_test_var, directory=''):
  
	filepath = directory + train_test_var + '/Inertial Signals/'
	filenames = sorted(os.listdir(filepath))

	# load X data (input)
	X = dataloader_group(filenames, filepath)
	# load output labels
	y = dataloader(directory + train_test_var + '/y_'+train_test_var+'.txt')
	return X, y

def load_labels():
	# load labels
	y_test = pd.read_csv('../Dataset/UCI HAR Dataset/test/y_test.txt', names=['Activity'], squeeze=True)
	y_labels = y_test.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS', 4:'SITTING', 5:'STANDING',6:'LAYING'})
	return y_labels