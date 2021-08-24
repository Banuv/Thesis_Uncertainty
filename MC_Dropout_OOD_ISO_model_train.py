from __future__ import print_function

import sys, os
import tensorflow as tf
import tensorflow.keras
import pandas as pd
import numpy as np
import sklearn
import keras
import csv

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

from keras_uncertainty.models import MCDropoutClassifier, MCDropoutRegressor
from keras_uncertainty.utils import numpy_regression_nll

def load_joint_space_csv_chunks(file_path):
    data_frame = pd.read_csv(file_path, skiprows=1, header=None)
    del data_frame[18]
    return data_frame

def load_task_space_csv_chunks(file_path):
    return pd.read_csv(file_path, skiprows=1, header=None)

def model_builder(input_shape, output_shape):
    def build_model(depth, width, reduction_factor):
        model = Sequential()

        for i in range(depth):
            num_neurons = max(int(width * (reduction_factor ** i)), 4)
            if i == 0:
                model.add(Dense(num_neurons, activation='relu', input_shape=(input_shape,)))
            else:
                model.add(Dense(num_neurons, activation='relu'))
                model.add(Dropout(0.5))

            model.add(BatchNormalization())

        model.add(Dense(output_shape, activation='sigmoid'))

        model.compile(loss='mse', optimizer='adam', metrics=["mae"])

        return model
    return build_model


def test_mcdropout_regressor(x_test_values, q_test_values, model, data_scaler):   
    mc_model = MCDropoutRegressor(model)
    inp = x_test_values  
    
    mean, std = mc_model.predict(inp, num_samples = 10)
    
    q_pred_unnormalised = data_scaler.inverse_transform(mean)
    
    q_sd_unnromalised = data_scaler.inverse_transform(std)
    
    global_mae = mean_absolute_error(q_test_values, mean)

    print("Testing MAE: {:.5f}".format(global_mae))

    return q_pred_unnormalised, q_sd_unnromalised
  




##please select the appropriate folder, willl use os.path.join() for completed script
TRAIN_FOLDER = '/home/dfki.uni-bremen.de/bmanickavasakan/newdataset_rh5_leg/leg_5steps/'
TEST_FOLDER = '/home/dfki.uni-bremen.de/bmanickavasakan/newdataset_rh5_leg/leg_5steps/test_4steps'

X_TRAIN_FILE = os.path.join(TRAIN_FOLDER, 'leg_forwardkinematics_x.csv')
Q_TRAIN_FILE = os.path.join(TRAIN_FOLDER, 'leg_sysstate_q.csv')
x_train = load_task_space_csv_chunks(X_TRAIN_FILE)
q_train = load_joint_space_csv_chunks(Q_TRAIN_FILE)

X_TEST_FILE = os.path.join(TEST_FOLDER, 'leg_forwardkinematics_x.csv')
Q_TEST_FILE = os.path.join(TEST_FOLDER, 'leg_sysstate_q.csv')
x_test = load_task_space_csv_chunks(X_TEST_FILE)
q_test = load_joint_space_csv_chunks(Q_TEST_FILE)



x_train_df = pd.DataFrame(x_train)
q_train_df = pd.DataFrame(q_train)
x_test_df = pd.DataFrame(x_test)
q_test_df = pd.DataFrame(q_test)


from sklearn.ensemble import IsolationForest


clf = IsolationForest(n_estimators=100, max_samples='auto', max_features=1, bootstrap=False, n_jobs= -1, random_state=42, verbose=0)
clf.fit(q_train_df)

pred = clf.predict(q_train_df)
q_train_df['anamoly'] = pred
print(q_train_df['anamoly'].value_counts())

InDistribution_Q_Train = q_train_df[q_train_df['anamoly'] == 1]
OutDistribution_Q_Train =   q_train_df[q_train_df['anamoly'] == -1]
InDistribution_X_Train =    x_train_df[q_train_df['anamoly'] == 1]
OutDistribution_X_Train =   x_train_df[q_train_df['anamoly'] == -1]

clf_test = IsolationForest(n_estimators=100, max_samples='auto', max_features=1, bootstrap=False, n_jobs= -1, random_state=42, verbose=0)
clf_test.fit(q_test_df)
pred_test = clf.predict(q_test_df)
q_test_df['anamoly'] = pred_test

InDistribution_Q_Test = q_test_df[q_test_df['anamoly'] == 1]
OutDistribution_Q_Test =q_test_df[q_test_df['anamoly'] == -1]
InDistribution_X_Test = x_test_df[q_test_df['anamoly'] == 1]
OutDistribution_X_Test =x_test_df[q_test_df['anamoly'] == -1]

x_train_1 = InDistribution_X_Train
q_train_1 = InDistribution_Q_Train.drop(['anamoly'], axis=1)
x_test_1 = InDistribution_X_Test
q_test_1 = InDistribution_Q_Test.drop(['anamoly'], axis=1)

OOD_x_train = OutDistribution_X_Train
OOD_q_train = OutDistribution_Q_Train.drop(['anamoly'], axis=1)
OOD_x_test = OutDistribution_X_Test
OOD_q_test = OutDistribution_Q_Test.drop(['anamoly'], axis=1)




x_scaler = MinMaxScaler()
q_scaler = MinMaxScaler()

#In order training set
x_train_1 = x_scaler.fit_transform(x_train_1)
q_train_1 = q_scaler.fit_transform(q_train_1)

#complete test set
x_test = x_scaler.transform(x_test)
#q_test = q_scaler.transform(q_test)

#split testing data
IOD_x_test = x_scaler.transform(x_test_1)
IOD_q_test = q_scaler.transform(q_test_1)

OOD_x_test = x_scaler.transform(OOD_x_test)
OOD_q_test = q_scaler.transform(OOD_q_test)

HYPERPARAMETERS = {'depth': 6, 'width': 64, 'reduction_factor':  1.1}

model = model_builder(9, 18)(**HYPERPARAMETERS)

#with tf.device('/gpu:2'):
hist = model.fit(x_train_1, q_train_1, epochs = 200, batch_size = 256, verbose = 1, validation_data=(IOD_x_test, IOD_q_test), use_multiprocessing=True, workers=1000)

model.save("MC_DROPOUT_OOD_SD_MODEL")


