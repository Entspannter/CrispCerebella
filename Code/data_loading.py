import pandas as pd
import os
from pandas import read_csv
from numpy import dstack

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

def load_labels(directory = '../Dataset/UCI HAR Dataset/test/y_test.txt'):
	# load labels
	y_test = pd.read_csv(directory, names=['Activity'], squeeze=True)
	y_labels = y_test.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS', 4:'SITTING', 5:'STANDING',6:'LAYING'})
	return y_labels