import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report
from numpy import unique

def create_heatmap(best_model, x_test, y_test, y_labels):
    # function to create a confusion matrix and a heatmap
    # create prediction based on the input dataset
    # convert the output from categorical to numerical
    prediction = np.argmax(best_model.predict(x_test), axis=-1)

    # reshape the predictions output
    prediction = prediction.reshape(prediction.shape[0], 1)

    # increment the output by 1 to match the y_test file
    prediction = prediction + 1

    # extract the unique labels and targets names of activities for the heatmap
    labels = unique(y_test)
    target_names = unique(y_labels)

    # create text report showing the main classification metrics
    clf_report = classification_report(y_test,
                                       prediction,
                                       labels,
                                       target_names,
                                       output_dict=True)

    # plot the report as a heatmap
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap="Blues")
    
    matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    
    print(matrix)
