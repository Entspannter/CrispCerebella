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

from data_loading import *
from test_functions import *

testX, testy = datasetloader('test', '../Dataset/UCI HAR Dataset/')
print("Shape of test Data:", testX.shape, testy.shape)

y_labels = load_labels()

# Heatmap for the LSTM model
BestLSTMModel = keras.models.load_model("../Models/Final_Model_LSTM0.8744485974311829")

HeatMap_ConfMatrix(BestLSTMModel, testX, testy, y_labels)

# Heatmap for the LSTM model
BestConvLSTMModel = keras.models.load_model("../Models/")

HeatMap_ConfMatrix(BestConvLSTMModel, testX, testy, y_labels)

# Heatmap for the LSTM model
BestCNNLSTMModel = keras.models.load_model("../Models/")

HeatMap_ConfMatrix(BestCNNLSTMModel, testX, testy, y_labels)



