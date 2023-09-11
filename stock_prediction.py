# File: stock_prediction.py
# Authors: Cheong Koo and Bao Vo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 25/07/2023 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

import numpy as np
import plotly.graph_objects as graphs
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf
import datetime
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer

def load_data(company, start_date, end_date, prediction_window=60, split=0.2, refresh=True, save=True, data_dir='data', feature_columns=['Open', 'High', 'Low', 'Close', 'Volume']):
	# Loads data from Yahoo Finance source, as well as scaling, normalizing and splitting.
    # Params:
	# 	company				(str)			: The company you want to train on, examples include AAPL, TESL, etc.
	#	start_date			(str)			: The start date for the data
	#	end_date			(str)			: The end date for the data
	#	prediction_window	(int)			: The historical sequence length used to predict, default is 60
	# 	split 				(str, float)	: Date string or float between 0 and 1 that defines split
	#	refresh				(bool)			: Whether to redownload data even if it exists, default is True
	#	save				(bool)			: Whether to save the data locally if it doesn't already exist, default is True
	#	data_dir			(str)			: Directory to store data, default is 'data'
	# 	feature_columns		(list)			: The list of features to use to feed into the model, default is everything grabbed from yahoo

	try:
		split_is_date = bool(datetime.strptime(split, '%Y-%m-%d'))
	except:
		split_is_date = False

	try:
		split_is_float = bool(0 < split < 1)
	except:
		split_is_float = False
	
	assert split_is_date or split_is_float, f'split must be a float between 0 and 1 or a string (\'yyyy-mm-dd\')'

	# Creates data directory if it doesn't exist
	if not os.path.isdir(data_dir):
		os.mkdir(data_dir)
		
	# Shorthand for provided data path and generated filename
	df_file_path = os.path.join(data_dir, f'{company}_{start_date}-{end_date}_{prediction_window}-window.csv')

	# Checks if data file with same data exists
	if os.path.exists(df_file_path) and not refresh:
		# If file exists and data shouldn't be updated, import as pandas data frame object
		# index_col='Date' makes the date the index rather than making a new coloumn
		df = pd.read_csv(df_file_path, parse_dates=True, index_col='Date')
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
	dataset['stock_df'] = df.copy()
	
	# Drop NaNs
	df.dropna(inplace=True)

	# For more details: 
	# https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html
	#------------------------------------------------------------------------------
	# Prepare Data
	## To do:
	# 1) Use a different price value eg. mid-point of Open & Close
	#------------------------------------------------------------------------------

	# If split is false, split the dataframe into training and testing data based on the split
	if split_is_float:
		# Convert the split percentage to a index
		train_start = int((1 - split) * len(df))
		# train contains all values before that index
		train = df[:train_start]
		# test contains all values after that index
		test = df[train_start:]
	# If split is a date, split the dataframe into training and testing based on that date
	else:
		# train contains all values before test_start_date
		train = df[df.index < split]
		# test contains all values after test_start_date
		test = df[df.index > split]
	# Add the train and test df to the dataset
	dataset['train_df'] = train
	dataset['test_df'] = test

	# Create dicts to store verions of each value for feature_columns
	column_scaler = {}
	column_x_train, column_y_train = {}, {}
	column_x_test, column_y_test = {}, {}
	column_model_inputs = {}

	for column in feature_columns:
		# Scaling each feature_column from 0 to 1
		# Note that, by default, feature_range=(0, 1). Thus, if you want a different 
		# 	feature_range (min,max) then you'll need to specify it here
		scaler = MinMaxScaler(feature_range=(0, 1))
		# Flatten and normalise the df
		# First, we reshape a 1D array(n) to 2D array(n,1)
		# We have to do that because sklearn.preprocessing.fit_transform()
		# requires a 2D array
		# Here n == len(test_scaled_data)
		# Then, we scale the whole array to the range (0,1)
		# The parameter -1 allows (np.)reshape to figure out the array size n automatically 
		# 	values.reshape(-1, 1) 
		# https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape'
		# When reshaping an array, the new shape must contain the same number of elements 
		# 	as the old shape, meaning the products of the two shapes' dimensions must be equal. 
		# When using a -1, the dimension corresponding to the -1 will be the product of 
		# 	the dimensions of the original array divided by the product of the dimensions 
		# 	given to reshape so as to maintain the same number of elements.
		test_scaled_data = scaler.fit_transform(train[column].values.reshape(-1, 1))
		# Save scalar used
		column_scaler[column] = scaler
		# Turn the 2D array back to a 1D array
		test_scaled_data = test_scaled_data[:,0]


		# Arrays for x and y data
		x_train, y_train = [], []

		# Prepare the training data
		for i in range(prediction_window, len(test_scaled_data)):
			# Offset x and y data by prediction_window
			x_train.append(test_scaled_data[i-prediction_window:i])
			y_train.append(test_scaled_data[i])
			
		# Convert them into an array
		x_train, y_train = np.array(x_train), np.array(y_train)
		# Now, x is a 2D array(p,q) where p = len(test_scaled_data) - prediction_window
		# and q = prediction_window; while y is a 1D array(p)

		# We now reshape x into a 3D array(p, q, 1); Note that x
		# 	is an array of p inputs with each input being a 2D array
		x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

		# Store the training data in the dicts under the name of the feature_column
		column_x_train[column], column_y_train[column] = x_train, y_train

		# Create the inputs which is the amount of prediction_window and all the test data
		model_inputs = df[column][len(df[column]) - len(test) - prediction_window:].values
		# Scale model_inputs using the previously used scaler
		model_inputs = scaler.transform(model_inputs.reshape(-1, 1))
		# Save model_inputs to dict to be added to the dataset
		column_model_inputs[column] = model_inputs

		# Create test data arrays
		x_test, y_test = [], []
		
		# Prepare the testing data
		for i in range(prediction_window, len(model_inputs)):
			# Offset x and y by prediction window
			x_test.append(model_inputs[i - prediction_window:i, 0])
			y_test.append(test_scaled_data[i])

		# Convert test data into numpy arrays
		x_test, y_test = np.array(x_test), np.array(y_test)
		# We now reshape x into a 3D array(p, q, 1); Note that x
		# 	is an array of p inputs with each input being a 2D array
		x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

		# Store the model_inputs in the dicts under the name of the feature_column
		column_x_test[column], column_y_test[column] = x_test, y_test

	# Add the MinMaxScaler instances to the dataset
	dataset['column_scaler'] = column_scaler
	# Add the training data instances to the dataset
	dataset['column_x_train'], dataset['column_y_train'] = column_x_train, column_y_train
	# Add the test data instances to the dataset
	dataset['column_x_test'], dataset['column_y_test'] = column_x_test, column_y_test
	# Add the model_inputs instances to the dataset
	dataset['column_model_inputs'] = column_model_inputs

	return dataset

def build_model(x_train, y_train, refresh=True, save=True, model_dir='model'):
	# Builds the model
	# Params:
	# 	x_train		(list)	: The x training data
	# 	y_train		(list)	: The y training data
	#	refresh 	(bool)	: Whether to retrain the model even if it exists, default is False
	#	save		(bool)	: Whether to save the model locally if it doesn't already exist, default is True
	#	model_dir	(str)	: Directory to store model, default is 'model'
	
	# Creates model directory if it doesn't exist
	if not os.path.isdir(model_dir):
		os.mkdir(model_dir)
	
	prediction_window = len(x_train[0])

	# Shorthand for provided model path and generated filename
	model_file_path = os.path.join(model_dir, f'model.keras_{prediction_window}-window')

	# Checks if data file with same data exists
	if os.path.exists(model_file_path) and not refresh:
		# If file exists and data shouldn't be updated, import as pandas data frame object
		# 'index_col=0' makes the date the index rather than making a new coloumn
		model = load_model(model_file_path)
		return model
	
	#------------------------------------------------------------------------------
	# Build the Model
	## TO DO:
	# 1) Change the model to increase accuracy?
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

	if save:
		model.save(model_file_path)

	return model

def predict_test(model, scaler, x_test, test_index, feature_columns=['Open', 'High', 'Low', 'Close', 'Volume']):
	# Uses model and data to make prediction
	# Params:
	# 	model					: The model previously generated to actually be tested
	# 	scaler					: The scaler used to process the data so it can be de-normalised
	# 	x_test					: The test data to be used for prediction
	#	test_index		(list)	: Datetime index of the test data to be added to prediction_df for graphs
	# 	feature_columns	(list)	: The list of features to use to feed into the model, default is everything grabbed from yahoo

	prediction_df = pd.DataFrame(index=test_index)

	#------------------------------------------------------------------------------
	# Make predictions on test data
	#------------------------------------------------------------------------------

	for column in feature_columns:
		predicted_prices = model.predict(x_test[column])
		# Clearly, as we transform our data into the normalized range (0,1),
		# we now need to reverse this transformation
		predicted_prices = scaler[column].inverse_transform(predicted_prices)
		predicted_prices = np.array(predicted_prices)
		prediction_df[column] = predicted_prices

	return prediction_df

def candlestick_plot(test_df, predicted_df, days=1, feature_columns=['Open', 'High', 'Low', 'Close', 'Volume']):
	# Uses model and data to make prediction
	# Params:
	# 	test_df			(df)	: The test data downloaded from yahoo
	# 	predicted_df	(df)	: The predictions based on the test data
	# 	days			(int)	: The amount of days to show in a candlestick, default is 1
	# 	feature_columns	(list)	: The list of features to use to feed into the model, default is everything grabbed from yahoo

	#------------------------------------------------------------------------------
	# Plot the test predictions
	## To do:
	# 1) Show chart of next few days (predicted)
	#------------------------------------------------------------------------------

	# Ensure that days is valid
	assert days >= 0, 'days must be bigger than or equal to 1'

	# Tell how the pd resample should treat the different vales
	aggregation = {
		'Open':'first',
		'High':'max',
		'Low':'min',
		'Close':'last',
		'Volume':'sum'
	}
	# Resample both the test data and the prediction data based on 'days'
	resampled_test_df = test_df.resample(f'{days}d').agg(aggregation)
	resampled_predicted_df = predicted_df.resample(f'{days}d').agg(aggregation)
	
	# Derrive the dates of of the graph from the dataset
	date_range = f'{test_df.index[0]:%b %Y} - {test_df.index[-1]:%b %Y}'

	# Create candlestick graph of test data
	test_data_graph = graphs.Candlestick(
		x=resampled_test_df.index,
		open=resampled_test_df['Open'],
		high=resampled_test_df['High'],
		low=resampled_test_df['Low'],
		close=resampled_test_df['Close'],
		showlegend=False
	)
	# Create scatter graphs of predicted data
	predicted_data_graphs = []
	for column in feature_columns:
		graph = graphs.Scatter(
			x=resampled_predicted_df.index,
			y=resampled_predicted_df[column],
			name=f'Predicted {column}'
		)
		predicted_data_graphs.append(graph)

	# Put graphs together in figure
	fig = graphs.Figure(data=([test_data_graph] + predicted_data_graphs))
	# Update figure titles
	fig.update_layout(
		title=f'{COMPANY} Share Prices {date_range}',
		yaxis_title='Price ($)'
	)

	return fig

def boxplot(test_df, predicted_df, days=1, feature_columns=['Open', 'High', 'Low', 'Close', 'Volume']):
	# Uses model and data to make prediction
	# Params:
	# 	test_df			(df)	: The test data downloaded from yahoo
	# 	predicted_df	(df)	: The predictions based on the test data
	# 	days			(int)	: The amount of days to show in a candlestick, default is 1
	# 	feature_columns	(list)	: The list of features to use to feed into the model, default is everything grabbed from yahoo

	#------------------------------------------------------------------------------
	# Plot the test predictions
	## To do:
	# 1) Show chart of next few days (predicted)
	#------------------------------------------------------------------------------

	# Ensure that days is valid
	assert days >= 0, 'days must be bigger than or equal to 1'

	# Tell how the pd resample should treat the different vales
	aggregation = {
		'Open':'first',
		'High':'max',
		'Low':'min',
		'Close':'last',
		'Volume':'sum'
	}
	# Resample both the test data and the prediction data based on 'days'
	resampled_test_df = test_df.resample(f'{days}d').agg(aggregation)
	resampled_predicted_df = predicted_df.resample(f'{days}d').agg(aggregation)
	
	# Derrive the dates of of the graph from the dataset
	date_range = f'{test_df.index[0]:%b %Y} - {test_df.index[-1]:%b %Y}'

	# Create box graph of test data
	test_data_graph = graphs.Box(
		x=resampled_test_df.index,
		y=resampled_test_df['Open'],
		showlegend=False
	)
	# Create scatter graphs of predicted data
	predicted_data_graphs = []
	for column in feature_columns:
		graph = graphs.Scatter(
			x=resampled_predicted_df.index,
			y=resampled_predicted_df[column],
			name=f'Predicted {column}'
		)
		predicted_data_graphs.append(graph)

	# Put graphs together in figure
	fig = graphs.Figure(data=([test_data_graph] + predicted_data_graphs))
	# Update figure titles
	fig.update_layout(
		title=f'{COMPANY} Share Prices {date_range}',
		yaxis_title='Price ($)'
	)

	return fig

def predict(model, scaler, model_inputs, prediction_days=60):
	# Uses model and data to make prediction
	# Params:
	# 	model					: The model previously generated to actually be tested
	# 	scaler					: The scaler used to process the data so it can be de-normalised
	# 	actual_prices	(list)	: The prices downloaded from yahoo to compare against
	# 	prediction_days	(int)	: How far ahead the final prediction should be, default is 1 (e.g in 60 days)

	#------------------------------------------------------------------------------
	# Predict next day
	#------------------------------------------------------------------------------


	real_data = [model_inputs[len(model_inputs) - prediction_days:, 0]]
	real_data = np.array(real_data)
	real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

	prediction = model.predict(real_data)
	prediction = scaler.inverse_transform(prediction)
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

	return prediction

if __name__ == '__main__':
	# Default params that must be set
	COMPANY = 'TSLA'
	START_DATE = '2015-01-01'
	END_DATE = '2023-12-31'
	CHOSEN_FEATURE = 'Close'
	REFRESH = False

	# Generate the dataset based on the company and the dates
	dataset = load_data(COMPANY, START_DATE, END_DATE, 120, 0.1, REFRESH)

	# Generate the model based on the training data
	model = build_model(dataset['column_x_train'][CHOSEN_FEATURE], dataset['column_y_train'][CHOSEN_FEATURE], REFRESH)

	# Make df of predictions to compare against test data
	prediction_df = predict_test(model, dataset['column_scaler'], dataset['column_x_test'], dataset['test_df'].index)

	# Show candlestick graph of prices
	candlestick_plot(dataset['test_df'], prediction_df, 7, [CHOSEN_FEATURE]).show()
	# Show boxplot of prices
	boxplot(dataset['test_df'], prediction_df, 7, [CHOSEN_FEATURE]).show()
	
	# Run the predictions based on the model and testing data
	prediction = predict(model, dataset['column_scaler'][CHOSEN_FEATURE], dataset['column_model_inputs'][CHOSEN_FEATURE])
	print(f'Prediction: {prediction}')