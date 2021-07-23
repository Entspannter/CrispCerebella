# file for all the data visualisation code
import pandas as pd
from numpy import unique
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import vstack
from CrispCerebella.Code.data_loading import *
from IPython.display import display


# In this section we will assign the classes (which correspond the motions 1-6)
# to each row of the 128 data points.
def rows_by_class(our_data, print_flag == True):
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
        if (print_flag == True)
        print(f'Class={i + 1}', f'total={data_grouped_size[i]}', f'percentage={class_percent}', sep="\t \t \t")
    return df_grouped


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
