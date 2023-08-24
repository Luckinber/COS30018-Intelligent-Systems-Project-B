# File: stock_prediction.py
# Authors: Cheong Koo and Bao Vo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 25/07/2023 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following:
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer
from collections import deque

def load_data(company, start_date, end_date, predict_window=50, save=True, refresh=True, data_dir='data', scale=True, shuffle=True, prediction_days=60, split_by_date=True,
                test_size=0.2, feature_columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']):
	# Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    # Params:
	# 	company			(str)	: The company you want to train on, examples include AAPL, TESL, etc.
	#	start_date		(str)	: The start date for the data
	#	end_date		(str)	: The end date for the data
	#	predict_window	(int)	: The historical sequence length used to predict, default is 50
	#	save			(bool)	: Whether to save the data locally if it doesn't already exist, default is True
	#	refresh			(boool)	: Whether to redownload data even if it exists, defualt is True
	#	data_dir		(str)	: Directory to store data, default is 'data'
	# 	scale 			(bool)	: Whether to scale prices from 0 to 1, default is True
	#	shuffle 		(bool)	: Whether to shuffle the dataset (both training & testing), default is True
	# 	prediction_days	(int)	: How far ahead the final prediction should be, default is 1 (e.g next day)
	# 	split_by_date 	(bool)	: whether we split the dataset into training/testing by date, setting it 
	# 		to False will split datasets in a random way
	# 	test_size 		(float)	: ratio for test data, default is 0.2 (20% testing data)
	# 	feature_columns	(list)	: the list of features to use to feed into the model, default is everything grabbed from yahoo_fin


	# Creates data directory if it doesn't exist
	if not os.path.isdir(data_dir):
		os.mkdir(data_dir)
		
	# Shorthand for provided data path and generated filename
	df_file_path = os.path.join(data_dir, f'{company}_{start_date}-{end_date}.csv')

	# Checks if data file with same data exists
	if os.path.exists(df_file_path) and not refresh:
		# If file exists and data shouldn't be updated, import as pandas data frame object
		# 'index_col=0' makes the date the index rather than making a new coloumn
		df = pd.read_csv(df_file_path, index_col=0)
	else:
		# If file doesn't exist, download data from yahoo
		df = yf.download(company, start=start_date, end=end_date, progress=False)
		# Save data to data file for loading next run if saving is on
		if save:
			df.to_csv(df_file_path)

	# make sure that the passed feature_columns exist in the dataframe
	for col in feature_columns:
		assert col in df.columns, f'{col} does noxt exist in the dataframe.'
	
	# Create dataset dictionary to store outputs
	dataset = {}
    # Store the original dataframe itself
	dataset['df'] = df.copy()

	# For more details:
	# https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html
	#------------------------------------------------------------------------------
	# Prepare Data
	## To do:
	# 1) Use a different price value eg. mid-point of Open & Close
	# 2) Change the Prediction days
	#------------------------------------------------------------------------------
	
	if scale:
		column_scaler = {}
		for column in feature_columns:
			scaler = MinMaxScaler(feature_range=(0, 1))
			# Note that, by default, feature_range=(0, 1). Thus, if you want a different 
			# 	feature_range (min,max) then you'll need to specify it here
			df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1)) 
			# Flatten and normalise the data
			# First, we reshape a 1D array(n) to 2D array(n,1)
			# We have to do that because sklearn.preprocessing.fit_transform() requires a 2D array
			# Here n == len(scaled_data)
			# Then, we scale the whole array to the range (0,1)
			# The parameter -1 allows (np.)reshape to figure out the array size n automatically 
			# 	values.reshape(-1, 1) 
			# https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape'
			# When reshaping an array, the new shape must contain the same number of elements 
			# 	as the old shape, meaning the products of the two shapes' dimensions must be equal. 
			# When using a -1, the dimension corresponding to the -1 will be the product of 
			# 	the dimensions of the original array divided by the product of the dimensions 
			# 	given to reshape so as to maintain the same number of elements.
			column_scaler[column] = scaler

		# Add the MinMaxScaler instances to the result returned
		dataset['column_scaler'] = column_scaler

	# Add the target column (label) by shifting by `prediction_days`
	df['future'] = df['Adj Close'].shift(-prediction_days)
	# last `prediction_days` columns contains NaN in future column
    # 	get them before droping NaNs
	last_sequence = np.array(df[feature_columns].tail(prediction_days))

	# Drop NaNs
	df.dropna(inplace=True)

	sequence_data = []
	sequences = deque(maxlen=predict_window)

	for entry, target in zip(df[feature_columns].values, df['future'].values):
		sequences.append(entry)
		if len(sequences) == predict_window:
			sequence_data.append([np.array(sequences), target])

	# Get the last sequence by appending the last `predict_window` sequence with `prediction_days` sequence
	# 	for instance, if predict_window=50 and prediction_days=10, last_sequence should be of 60 (that is 50+10) length
	# this last_sequence will be used to predict future stock prices that are not available in the dataset
	last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
	last_sequence = np.array(last_sequence).astype(np.float32)
	# add to dataset
	dataset['last_sequence'] = last_sequence

	# To store the training and test data
	x, y = [], []
	for seq, target in sequence_data:
		x.append(seq)
		y.append(target)

	# Convert to numpy arrays
	x = np.array(x)
	y = np.array(y)

	if split_by_date:
		# Split the dataset into training and testing sets by date
		train_samples = int((1 - test_size) * len(x))
		dataset['x_train'] = x[:train_samples]
		dataset['y_train'] = y[:train_samples]
		dataset['x_test']  = x[train_samples:]
		dataset['y_test']  = y[train_samples:]
		# if shuffle:
		# 	# Shuffle the dataset for training
		# 	shuffle_in_unison(dataset['x_train'], dataset['y_train'])
		# 	shuffle_in_unison(dataset['x_test'], dataset['y_test'])
	else:
		# Split dataset randomly
		dataset['x_train'], dataset['y_train'], dataset['x_test'], dataset['y_test'] = train_test_split(x, y, test_size=test_size, shuffle=True)
	
	# # Get the list of test set dates
	# dates = dataset['x_test'][:, -1, -1]
	# # Retrieve test features from the original dataframe
	# dataset['test_df'] = dataset['df'].loc[Dates]
	# # Remove duplicated dates in the testing dataframe
	# dataset['test_df'] = dataset["test_df"][~dataset['test_df'].index.duplicated(keep='first')]
	# # Remove dates from the training/testing sets & convert to float32
	# dataset['x_train'] = dataset['x_train'][:, :, :len(feature_columns)].astype(np.float32)
	# dataset['x_test'] = dataset['x_test'][:, :, :len(feature_columns)].astype(np.float32)

	return dataset

def build_model(x_train, y_train):
	#------------------------------------------------------------------------------
	# Build the Model
	## TO DO:
	# 1) Check if data has been built before. 
	# If so, load the saved data
	# If not, save the data into a directory
	# 2) Change the model to increase accuracy?
	#------------------------------------------------------------------------------
	model = Sequential() # Basic neural network
	# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
	# for some useful examples

	model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
	# This is our first hidden layer which also spcifies an input layer. 
	# That's why we specify the input shape for this layer; 
	# i.e. the format of each training example
	# The above would be equivalent to the following two lines of code:
	# model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
	# model.add(LSTM(units=50, return_sequences=True))
	# For som eadvances explanation of return_sequences:
	# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
	# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
	# As explained there, for a stacked LSTM, you must set return_sequences=True 
	# when stacking LSTM layers so that the next LSTM layer has a 
	# three-dimensional sequence input. 

	# Finally, units specifies the number of nodes in this layer.
	# This is one of the parameters you want to play with to see what number
	# of units will give you better prediction quality (for your problem)

	model.add(Dropout(0.2))
	# The Dropout layer randomly sets input units to 0 with a frequency of 
	# rate (= 0.2 above) at each step during training time, which helps 
	# prevent overfitting (one of the major problems of ML). 

	model.add(LSTM(units=50, return_sequences=True))
	# More on Stacked LSTM:
	# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

	model.add(Dropout(0.2))
	model.add(LSTM(units=50))
	model.add(Dropout(0.2))

	model.add(Dense(units=1)) 
	# Prediction of the next closing value of the stock price

	# We compile the model by specify the parameters for the model
	# See lecture Week 6 (COS30018)
	model.compile(optimizer='adam', loss='mean_squared_error')
	# The optimizer and loss are two important parameters when building an 
	# ANN model. Choosing a different optimizer/loss can affect the prediction
	# quality significantly. You should try other settings to learn; e.g.
		
	# optimizer='rmsprop'/'sgd'/'adadelta'/...
	# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

	# Now we are going to train this model with our training data 
	# (x_train, y_train)
	model.fit(x_train, y_train, epochs=25, batch_size=32)
	# Other parameters to consider: How many rounds(epochs) are we going to 
	# train our model? Typically, the more the better, but be careful about
	# overfitting!
	# What about batch_size? Well, again, please refer to 
	# Lecture Week 6 (COS30018): If you update your model for each and every 
	# input sample, then there are potentially 2 issues: 1. If you training 
	# data is very big (billions of input samples) then it will take VERY long;
	# 2. Each and every input can immediately makes changes to your model
	# (a souce of overfitting). Thus, we do this in batches: We'll look at
	# the aggreated errors/losses from a batch of, say, 32 input samples
	# and update our model based on this aggregated loss.

	# TO DO:
	# Save the model and reload it
	# Sometimes, it takes a lot of effort to train your model (again, look at
	# a training data with billions of input samples). Thus, after spending so 
	# much computing power to train your model, you may want to save it so that
	# in the future, when you want to make the prediction, you only need to load
	# your pre-trained model and run it on the new input for which the prediction
	# need to be made.

	return model

def parse_test_data(company, start_date, end_date, data_dir, price_value, prediction_days, scaler):
	#------------------------------------------------------------------------------
	# Test the model accuracy on existing data
	#------------------------------------------------------------------------------
	
	# Load the test data
	test_data = load_data(company, start_date, end_date, data_dir)

	# The above bug is the reason for the following line of code
	test_data = test_data[1:]

	actual_prices = test_data[price_value].values

	total_dataset = pd.concat((data[price_value], test_data[price_value]), axis=0)

	model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
	# We need to do the above because to predict the closing price of the fisrt
	# prediction_days of the test period [TEST_START, TEST_END], we'll need the 
	# data from the training period

	model_inputs = model_inputs.reshape(-1, 1)
	# TO DO: Explain the above line

	model_inputs = scaler.transform(model_inputs)
	# We again normalize our closing price data to fit them into the range (0,1)
	# using the same scaler used above 
	# However, there may be a problem: scaler was computed on the basis of
	# the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
	# but there may be a lower/higher price during the test period 
	# [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
	# greater than one)
	# We'll call this ISSUE #2

	# TO DO: Generally, there is a better way to process the data so that we 
	# can use part of it for training and the rest for testing. You need to 
	# implement such a way

	return model_inputs, actual_prices

def predict(prediction_days, model_inputs, model, scaler, actual_prices):
	#------------------------------------------------------------------------------
	# Make predictions on test data
	#------------------------------------------------------------------------------
	x_test = []
	for x in range(prediction_days, len(model_inputs)):
		x_test.append(model_inputs[x - prediction_days:x, 0])

	x_test = np.array(x_test)
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
	# TO DO: Explain the above 5 lines

	predicted_prices = model.predict(x_test)
	predicted_prices = scaler.inverse_transform(predicted_prices)
	# Clearly, as we transform our data into the normalized range (0,1),
	# we now need to reverse this transformation 
	#------------------------------------------------------------------------------
	# Plot the test predictions
	## To do:
	# 1) Candle stick charts
	# 2) Chart showing High & Lows of the day
	# 3) Show chart of next few days (predicted)
	#------------------------------------------------------------------------------

	plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
	plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
	plt.title(f"{COMPANY} Share Price")
	plt.xlabel("Time")
	plt.ylabel(f"{COMPANY} Share Price")
	plt.legend()
	plt.show()

	#------------------------------------------------------------------------------
	# Predict next day
	#------------------------------------------------------------------------------


	real_data = [model_inputs[len(model_inputs) - prediction_days:, 0]]
	real_data = np.array(real_data)
	real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

	prediction = model.predict(real_data)
	prediction = scaler.inverse_transform(prediction)
	print(f"Prediction: {prediction}")

	# A few concluding remarks here:
	# 1. The predictor is quite bad, especially if you look at the next day 
	# prediction, it missed the actual price by about 10%-13%
	# Can you find the reason?
	# 2. The code base at
	# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
	# gives a much better prediction. Even though on the surface, it didn't seem 
	# to be a big difference (both use Stacked LSTM)
	# Again, can you explain it?
	# A more advanced and quite different technique use CNN to analyse the images
	# of the stock price changes to detect some patterns with the trend of
	# the stock price:
	# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
	# Can you combine these different techniques for a better prediction??

if __name__ == '__main__':
	COMPANY = 'TSLA'
	TRAIN_START = '2015-01-01'
	TRAIN_END = '2020-01-01'
	PRICE_VALUE = 'Close'
	TEST_START = '2020-01-02'
	TEST_END = '2022-12-31'
	PREDICTION_DAYS = 60
	dataset = load_data(COMPANY, TRAIN_START, TRAIN_END)
	model = build_model(dataset['x_train'], dataset['y_train'])
	print()
	# model_inputs, actual_prices = parse_test_data(COMPANY, TEST_START, TEST_END, 'data', PRICE_VALUE, PREDICTION_DAYS, scaler)
	# predict(PREDICTION_DAYS, model_inputs, model, scaler, actual_prices)