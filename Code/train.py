# Import own functions and model-classes
from train_functions import *
from data_loading import *
from architecture import *

# load all train
trainX, trainy = datasetloader('train', '../Dataset/UCI HAR Dataset/')
print("Shape of training Data:", trainX.shape, trainy.shape)
# load all test
testX, testy = datasetloader('test', '../Dataset/UCI HAR Dataset/')
print("Shape of test Data:", testX.shape, testy.shape)

# transform the output date into categorical (one-hot-encoding) and print the updated shapes
trainX, trainy, testX, testy = transform_dataset(trainX, trainy, testX, testy)

n_timesteps, n_features, N_OUTPUTS = trainX.shape[1], trainX.shape[2], trainy.shape[1]

n_steps, n_length = 4, 32

# define the inout share for the LSTM model
INPUT_SHAPE = (n_timesteps, n_features)

# define input variable for the k-fold CV and the model estimation functions
Num_Epochs_Search = 5
Num_Epochs_Fit = 5
Max_Trials = 3
Num_K_Folds = 5
Model_Name_LSTM = "LSTM"
batch_size = 64

# merge the train and test data to recreate the initial dataset for later
# splitting in the k-fold process
X_LSTM = np.concatenate((trainX, testX), axis=0)
y_LSTM = np.concatenate((trainy, testy), axis=0)

# Hyperparameters tuning inside of k-fold CV
Best_HP_LSTM, Best_Epoch_LSTM = k_fold_cv(X_LSTM,
                                          y_LSTM,
                                          LSTMStacked,
                                          Model_Name_LSTM,
                                          INPUT_SHAPE,
                                          N_OUTPUTS,
                                          Max_Trials,
                                          Num_Epochs_Search,
                                          Num_Epochs_Fit,
                                          Num_K_Folds)

# Print the hyperparameters that yielded the best model all over the k-folds
print(Best_HP_LSTM.get_config()['values'])

# Build a new model with the HP, train it with the initial training set and evaluate it with the test set
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

# Save the final best model
BestModelLSTM.save("../Models/Final_Model_LSTM" + str(round(LSTM_accuracy, 4)*100))
