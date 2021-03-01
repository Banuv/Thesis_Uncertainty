#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function

import sys, os

import tensorflow.keras
import pandas as pd
import numpy as np
import sklearn
import keras
import csv

from sklearn.preprocessing import MinMaxScaler
from load_rh5_dataset import load_joint_space_csv_chunks, load_task_space_csv_chunks
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def split_OOD_Data():

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

	# In[1]:

	stats_q_train = pd.DataFrame()
	stats_q_train["Mean"] = q_train.mean()
	stats_q_train["Var"] = q_train.var()
	stats_q_train["STD"] = q_train.std()
	stats_q_train["OneSigmaMax"] = stats_q_train["Mean"] + stats_q_train["STD"]
	stats_q_train["OneSigmaMin"] = stats_q_train["Mean"] - stats_q_train["STD"]
	stats_q_train.T

	# In[2]:

	max_std = stats_q_train["STD"].max()
	colomn_max_std = stats_q_train["STD"].idxmax()

	# In[3]:

	max = stats_q_train.loc[colomn_max_std, "Mean"] + max_std
	min = stats_q_train.loc[colomn_max_std, "Mean"] - max_std

	# In[4]:

	InDistribution_Q_Train = q_train[q_train[colomn_max_std].le(max) & q_train[colomn_max_std].ge(min)]
	OutDistribution_Q_Train = q_train[q_train[colomn_max_std].ge(max) | q_train[colomn_max_std].le(min)]
	InDistribution_X_Train = x_train[q_train[colomn_max_std].le(max) & q_train[colomn_max_std].ge(min)]
	OutDistribution_X_Train = x_train[q_train[colomn_max_std].ge(max) | q_train[colomn_max_std].le(min)]

	# In[5]
	InDistribution_Q_Test = q_test[q_test[colomn_max_std].le(max) & q_test[colomn_max_std].ge(min)]
	OutDistribution_Q_Test = q_test[q_test[colomn_max_std].ge(max) | q_test[colomn_max_std].le(min)]
	InDistribution_X_Test = x_test[q_test[colomn_max_std].le(max) & q_test[colomn_max_std].ge(min)]
	OutDistribution_X_Test = x_test[q_test[colomn_max_std].ge(max) | q_test[colomn_max_std].le(min)]
	
	a = [InDistribution_X_Train, InDistribution_Q_Train, OutDistribution_X_Train, OutDistribution_Q_Train] 		b = [InDistribution_X_Test, InDistribution_Q_Test, OutDistribution_X_Test, OutDistribution_Q_Test]
	return a, b


