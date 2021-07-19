
#pip install -q -U keras-tuner

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

from train_functions import *
from data_loading import *
from architecture import *


# load all train
trainX, trainy = datasetloader('train', '../Dataset/UCI HAR Dataset/')
print("Shape of training Data:",trainX.shape, trainy.shape)
# load all test
testX, testy = datasetloader('test', '../Dataset/UCI HAR Dataset/')
print("Shape of test Data:",testX.shape, testy.shape)

trainX, trainy, testX, testy = transform_dataset(trainX, trainy, testX, testy)

n_timesteps, n_features, N_OUTPUTS = trainX.shape[1], trainX.shape[2], trainy.shape[1]

n_steps, n_length = 4, 32

INPUT_SHAPE=(n_timesteps,n_features)

Num_Epochs_Search = 5
Num_Epochs_Fit = 5
Max_Trials = 3
Num_K_Folds = 5
Model_Name_LSTM = "LSTM"
batch_size = 64

X_LSTM = np.concatenate((trainX, testX),axis=0)
y_LSTM = np.concatenate((trainy, testy),axis=0)

Best_HP_LSTM, Best_Epoch_LSTM= K_Fold_CV(X_LSTM,
                                                              y_LSTM,
                                                              LSTMStacked,
                                                              Model_Name_LSTM,
                                                              INPUT_SHAPE,
                                                              N_OUTPUTS,
                                                              Max_Trials,
                                                              Num_Epochs_Search,
                                                              Num_Epochs_Fit,
                                                              Num_K_Folds)

Best_HP_LSTM.get_config()['values']

# Build a new model with the HP, train it with the inital training set and eveluate it with the test set 

BestModelLSTM, LSTM_accuracy = evaluate_model(trainX,
                               trainy,
                               testX,
                               testy,
                               INPUT_SHAPE,
                               N_OUTPUTS,
                               Best_HP_LSTM,
                               LSTMStacked,
                               Best_Epoch_LSTM,
                               batch_size)

BestModelLSTM.save("../Models/Final_Model_LSTM")