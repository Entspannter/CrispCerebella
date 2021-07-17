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


# load the dataset, returns train and test X and y elements
def transform_dataset(trainX, trainy, testX, testy):
	print(trainX.shape, trainy.shape)
	print(testX.shape, testy.shape)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy


def HP_Search(hypermodel ,MaxTrial, Num_Epochs_Search, Train_X, Train_y, ID):
# Define the tuner 
  tuner = keras_tuner.BayesianOptimization(
    hypermodel,
    objective = 'val_accuracy',
    max_trials = MaxTrial,
    tuner_id= ID,
    overwrite = True
)
  # Define an early stop 
  es = EarlyStopping(monitor='val_accuracy',min_delta=0.01, mode='max', verbose = 2, patience = 5)

  # Start the search of hyperparemeters
  tuner.search(Train_X, Train_y, epochs = Num_Epochs_Search, validation_split = 0.2,callbacks=[es])

  # Get the tuned hyperparemters 
  best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
  print("[The tuned hyperparemters]:", best_hps.get_config()['values'])

  # build a new model with the best hyperparemters
  model = tuner.hypermodel.build(best_hps)

  return model, best_hps

def K_Fold_CV (X, y, Model_Class, Model_Name, Input_Shape, Num_Outputs, Max_Trials, Num_Epochs_Search, Num_Epochs_Fit, Num_K_Folds):

  kf = KFold(n_splits = Num_K_Folds, random_state=None)
  acc_score = []
  k = 0
  for train_index , test_index in kf.split(X):
      print("Statring fold number {}". format(k+1))
      X_train , X_test = X[train_index,:],X[test_index,:]
      y_train , y_test = y[train_index] , y[test_index]

      hypermodel = Model_Class(Input_Shape, Num_Outputs)

      Best_Model, best_hp = HP_Search(hypermodel, Max_Trials, Num_Epochs_Search, X_train, y_train, "Tuner_"+str(Model_Name))

      history = Best_Model.fit(X_train, y_train, epochs = Num_Epochs_Fit, validation_split = 0.2)

      val_acc_per_epoch = history.history['val_accuracy']
      best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
      print('Best epoch: %d' % (best_epoch,))

      eval_result = Best_Model.evaluate(X_test, y_test)

      test_acc = eval_result[1]

      if k != 0:
        if test_acc >= max(acc_score):
          print("{} > {} ". format(test_acc,max(acc_score)))
          Best_HP = best_hp
          Best_Epoch = best_epoch
          print("Overriding the HP")
      else:
        Best_HP = best_hp
        Best_Epoch = best_epoch
  
      acc_score.append(test_acc)
      print("Fold number {} is done". format(k+1))
      k += 1
  
  avg_acc_score = sum(acc_score)/Num_K_Folds

  print('Accuracy of each fold - {}'.format(acc_score))
  print('Avg accuracy : {}'.format(avg_acc_score))

  return Best_HP, Best_Epoch

  

def evaluate_model(trainX, trainy, testX, testy, Input_Shape, Num_Outputs, Best_HP, Model_Class, Num_Epochs_Fit, Batch_Size):

	# create a new instace of the model class
	model = Model_Class(Input_Shape, Num_Outputs)

	# Build a new model with the best hyperparameters
	Best_Model = model.build(Best_HP)
 
	# Fit the model 
	history = Best_Model.fit(trainX, trainy, epochs = Num_Epochs_Fit, validation_split = 0.2)
 
	# evaluate model
	_, accuracy = Best_Model.evaluate(testX, testy, batch_size = Batch_Size, verbose=0)
 
  # summarize history for accuracy
	plt.plot(history.history['accuracy'],color = "#0000e6")
	plt.plot(history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'],color ="#0000e6")
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
 
	print("The accureay of the model is {}". format(accuracy))
 
	return Best_Model, accuracy