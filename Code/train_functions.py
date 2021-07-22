import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import keras_tuner
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
import tensorflow as tf
import os
import numpy as np
import random


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


# load the dataset, returns train and test X and y elements
def transform_dataset(x_train, y_train, x_test, y_test):
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    # zero-offset class values
    y_train = y_train - 1
    y_test = y_test - 1
    # one hot encode y
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def hp_search(hypermodel, max_trials, num_epochs_search, x_train, y_train, tuner_id):
    # Define the tuner
    tuner = keras_tuner.BayesianOptimization(
        hypermodel,
        objective='val_accuracy',
        max_trials=max_trials,
        tuner_id=tuner_id,
        overwrite=True
    )
    # Define an early stop
    es = EarlyStopping(monitor='val_accuracy', min_delta=0.01, mode='max', verbose=2, patience=5)

    # Start the search of hyperparemeters
    tuner.search(x_train, y_train, epochs=num_epochs_search, validation_split=0.2, callbacks=[es])

    # Get the tuned hyperparemters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # build a new model with the best hyperparemters
    model = tuner.hypermodel.build(best_hps)

    return model, best_hps


def k_fold_cv(x, y, model_class, model_name, input_shape, num_outputs, max_trials, num_epochs_search, num_epochs_fit,
              num_k_folds):
    # provides train/test indices to split data in train/test sets.
    kf = KFold(n_splits=num_k_folds, random_state=None)

    acc_score = []
    k = 0

    for train_index, test_index in kf.split(x):
        print("Starting fold number {}".format(k + 1))
        # split dataset into k consecutive folds based on the generated train/test indices
        X_train, X_test = x[train_index, :], x[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        # instantiate the hypermodel from the chosen model class (StackedLSTM/CNNLSTM/ConvLSTM)
        hypermodel = model_class(input_shape, num_outputs)

        # start search for the best hyperparemeters and the corresponding model
        best_model, best_hp = hp_search(hypermodel, max_trials, num_epochs_search, X_train, y_train,
                                        "Tuner_" + str(model_name))

        # fit the model
        history = best_model.fit(X_train, y_train, epochs=num_epochs_fit, validation_split=0.2)

        val_acc_per_epoch = history.history['val_accuracy']

        # find the optimal number of epochs to train the model 
        # with the hyperparameters obtained from the search.
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        # re-instantiate the hypermodel and train it with the optimal number of epochs from above
        hypermodel_new = model_class(input_shape, num_outputs)
        new_best_model = hypermodel_new.build(best_hp)
        new_best_model.fit(X_train, y_train, epochs=best_epoch, validation_split=0.2)

        # evaluate the model 
        eval_result = new_best_model.evaluate(X_test, y_test)

        # extract the accuracy of the model 
        test_acc = eval_result[1]

        # for each fold, save the hyperparameters that generate the model with the highest accuracy.
        # after the first fold, check if the accuracy for the best model for the current fold has
        # outperformed the previous fold; if yes, overwrite the best hyperparameters and number
        # of optical epochs for fitting
        if k > 0:
            if test_acc > max(acc_score):
                print("{} > {} ".format(test_acc, max(acc_score)))
                print("Overriding the HP")
                besthp = best_hp
                bestepochs = best_epoch
        else:
            besthp = best_hp
            bestepochs = best_epoch

        # save the accuracy of the best model of the current fold into a list
        acc_score.append(test_acc)
        print("Fold number {} is done".format(k + 1))
        k += 1

    # calculate the mean accuracy of all k folds.
    avg_acc_score = sum(acc_score) / num_k_folds

    print('Accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))

    return besthp, bestepochs


def evaluate_model(x_train, y_train, x_test, y_test, input_shape, num_outputs, best_hp, model_class, best_epochs,
                   batch_size):

    # create a new instance of the model class
    model = model_class(input_shape, num_outputs)

    # Build a new model with the best hyperparameters
    best_model = model.build(best_hp)

    # Fit the model
    history = best_model.fit(x_train, y_train, epochs=best_epochs, validation_split=0.2)

    # evaluate model
    _, accuracy = best_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

    # summarize history for accuracy
    plt.plot(history.history['accuracy'], color="#0000e6")
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'], color="#0000e6")
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    print("The accuracy of the model is {}".format(accuracy))

    return best_model, accuracy
