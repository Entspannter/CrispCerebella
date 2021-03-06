# CrispCerebella
#### Deep Learning Project, Summer Semester 2021, HPI Institute  

<p align="center">
  <img src="https://user-images.githubusercontent.com/56883449/126058385-eb816667-79ce-4f4a-a6ed-e791f960bee4.png" alt="Sublime's custom image"/>
</p>

**Table of Contents**

[Overview](#Overview)

[Files](#Files)

[Dataset](#Dataset)

[Requirements](#Requirements)

[Links](#Links)

[Used Materials](#Used_Materials)

[References](#References)


# Overview
This repository provides the codes and data used for our Deep Learning Project - "Human Activity Recognition from Sensor Data".
The goal of our project is to build a model which is able to classify and predict human activities performed by users based on accelerometer and gyroscope sensor data, which were collected by a smartphone located on the user's waist. 

The activities to be classified are:
1. WALKING
2. WALKING_UPSTAIRS
3. WALKING_DOWNSTAIRS
4. SITTING
5. STANDING
6. LAYING

To achieve our goal we implemented and evaluated several machine learning (ML) approaches. We built four different ML: Stacked LSTM with two layers, ConvLSTM, CNN LSTM, and Transfer Learning. In addition, we used hyperparameter tuning and K-fold Cross Validation to optimize the performance of the models, and eventually choose the best model with the best accuracy.

# Files
Our repository is divided into 4 folders as follows:
1. Code - Contains the scripts used in our project 
2. Dataset - Contains the UCI dataset files
3. Models - Contains our trained ML models
4. WISDM - data set for Transfer Learning

Adittional files:

**Colab demo** - a goolge colab demonstrating the most important analysis steps and results.  

# Dataset
HAR Dataset from UCI dataset was used. This dataset was collected from 30 subjects (between the ages 19-48 years old) performing different activities with a smartphone (Samsung Galaxy S II) attached to their waist. The accelerometer and Gyroscope were already embedded to the smartphone. The data obtained was preprocessed and labeled. The time signals were preprocessed using a noise filter and later sampled in sliding windows of 2.56 seconds and 50\% overlap (128 readings/window). In addition, the accelerometer data was divided into gravitational and body motion sets with a Butterworth low-pass filter into body acceleration and gravity. For the gravitational force, a filter with a 0.3 Hz cut-off frequency was used. Finally, a 561 vector was available for each window by calculating variables of time and frequency. Finally, the dataset was divided into two sets (70% - Training data, 30% - Test data) [1]. 

# Requirements
<ul>
<li><a href="https://github.com/fchollet/keras">Keras</a> (Recommended version 2.4.0)</li>
<li><a href="https://www.tensorflow.org/" rel="nofollow">Tensorflow</a> (Recommended version 2.5.0)</li>
<li><a href="https://www.python.org/" rel="nofollow">Python 3.8</a></li>
<li><a href="http://scikit-learn.org/stable/" rel="nofollow">Scikit-learn</a></li>
<li>Other used libraries: Numpy, Matplotlib, Seaborn, sklearn.metrics, Scipy, OS, and Pandas</li>
</ul>

# Links
[Link to Dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)

# Used_Materials

The following sourses were used as a resource of information for creating the models and optimizating them:

1. Brownlee, Jason. n.d. ???LSTMs for Human Activity Recognition Time Series Classification.??? Accessed July 23, 2021. https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/.
2. Brownlee, Jason. 2018. ???1D Convolutional Neural Network Models for Human Activity Recognition.??? September 21, 2018. https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/.
3. Prost, Julie. 2020. ???How to Perform Hyperparameter Tuning with Keras Tuner | Sicara.??? November 26, 2020. https://www.sicara.ai/blog/hyperparameter-tuning-keras-tuner.
4. ???Einf??hrung in Den Keras Tuner  |  TensorFlow Core.??? n.d. Accessed July 23, 2021. https://www.tensorflow.org/tutorials/keras/keras_tuner.
5. ???Modelle Speichern Und Laden  |  TensorFlow Core.??? n.d. Accessed July 23, 2021. https://www.tensorflow.org/tutorials/keras/save_and_load.
6. Nils. 2018. ???Human Activity Recognition (HAR) Tutorial with Keras and Core ML (Part 1) | by Nils | Towards Data Science.??? August 9, 2018. https://towardsdatascience.com/human-activity-recognition-har-tutorial-with-keras-and-core-ml-part-1-8c05e365dfa0.
7. Benner, Jonas. 2020. ???Cross-Validation and Hyperparameter Tuning: How to Optimise Your Machine Learning Model | by Jonas Benner | Aug, 2020 | Medium | Towards Data Science.??? August 6, 2020. https://towardsdatascience.com/cross-validation-and-hyperparameter-tuning-how-to-optimise-your-machine-learning-model-13f005af9d7d.


# References 
[1] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013. 

