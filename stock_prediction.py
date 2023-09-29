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
import tensorflow as tf
import yfinance as yf
import os

from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, InputLayer, LSTM, SimpleRNN, GRU
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

def load_data(company, start_date, end_date, refresh=True, save=True, data_dir='data'):
	# Loads data from Yahoo Finance source.
    # Params:
	# 	company		(str)	: The company you want to train on, examples include AAPL, TESL, etc.
	#	start_date	(str)	: The start date for the data
	#	end_date	(str)	: The end date for the data
	#	refresh		(bool)	: Whether to redownload data even if it exists, default is True
	#	save		(bool)	: Whether to save the data locally if it doesn't already exist, default is True
	#	data_dir	(str)	: Directory to store data, default is 'data'
	# Returns:
	# 	dataset		(dict)	: The dataset dictionary containing the df

	# Creates data directory if it doesn't exist
	if not os.path.isdir(data_dir):
		os.mkdir(data_dir)
		
	# Shorthand for provided data path and generated filename
	df_file_path = os.path.join(data_dir, f'{company}_{start_date}-{end_date}.csv')

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
	
	# Drop NaNs
	df.dropna(inplace=True)

	return df

def prepare_data(df, sequence_length=60, split=0.2, feature_columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']):
	# Scales, normalizes and splits the data into training and testing data
	# Params:
	# 	df				(df)		: The dataframe to be processed
	# 	sequence_length	(int)		: The historical sequence length used to predict
	# 	split			(float, str): The percentage of data to be used for testing or the date to split the data on
	# 	feature_columns	(list)		: The list of features to use to feed into the model, default is everything grabbed from yahoo
	# Returns:
	# 	dataset			(dict)		: The dataset dictionary containing the df, train_df, test_df, scaler, x_train, y_train, x_test, y_test, model_inputs

	# make sure that the passed feature_columns exist in the dataframe
	for column in feature_columns:
		assert column in df.columns, f'{column} does noxt exist in the dataframe.'

	# Determine whether split is a date or a percentage
	split_is_date = isinstance(split, str) and isinstance(datetime.strptime(split, '%Y-%m-%d'), datetime)
	split_is_float = isinstance(split, float) and 0 < split < 1

	# Ensure that split is valid
	assert split_is_date or split_is_float, f'split must be a float between 0 and 1 or a string (\'yyyy-mm-dd\')'

	# If split is false, split the dataframe into training and testing data based on the split
	if split_is_float:
		# Convert the split percentage to an index
		train_start = int((1 - split) * len(df))
		# train_df contains all values before that index
		train_df = df[:train_start]
		# test_df contains all values after that index
		test_df = df[train_start:]
	# If split is a date, split the dataframe into training and testing based on that date
	else:
		# train_df contains all values before test_start_date
		train_df = df[df.index < split]
		# test_df contains all values after test_start_date
		test_df = df[df.index > split]

	# Scale the data
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled_data = scaler.fit_transform(df[feature_columns].values)

	# Prepare training data
	x_train, y_train = [], []
	for i in range(sequence_length, len(train_df)):
		# Add the previous sequence_length values to x_train
		x_train.append(scaled_data[i - sequence_length: i])
		# Add the current value to y_train
		y_train.append(scaled_data[i])

	# Prepare testing data
	x_test, y_test = [], []
	for i in range(len(train_df), len(df)):
		# Add the previous sequence_length values to x_test
		x_test.append(scaled_data[i - sequence_length: i])
		# Add the current value to y_test
		y_test.append(scaled_data[i])

	# Convert them into numpy arrays
	x_train, y_train = np.array(x_train), np.array(y_train)
	x_test, y_test = np.array(x_test), np.array(y_test)

	# Create model inputs from the last sequence_length values from the original dataset
	model_inputs = x_test[-1]

	# Reshape the model_inputs to be 3-dimensional in the form [number of samples, sequence length, number of features]
	model_inputs = np.reshape(model_inputs, (1, sequence_length, len(feature_columns)))

	# Create dataset dictionary
	dataset = {
		'stock_df': df.copy(),
		'train_df': train_df,
		'test_df': test_df,
		'scaler': scaler,
		'x_train': x_train,
		'y_train': y_train,
		'x_test': x_test,
		'y_test': y_test,
		'model_inputs': model_inputs
	}

	return dataset

def create_model(input_shape, cell=LSTM, layer_size=[50, 50, 50], dropout=0.2, optimizer='adam', loss='mean_squared_error', feature_columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']):
	# Builds the model
	# Params:
	# 	input_shape	(list)	: The shape of the input data
	# 	cell		(func)	: The type of cell to use in the model, default is LSTM
	# 	layer_size	(list)	: The size of each layer in the model, default is [50, 50, 50]
	# 	dropout		(float)	: The dropout rate to use in the model, default is 0.2
	# 	optimizer	(str)	: The optimizer to use in the model, default is 'adam'
	# 	loss		(str)	: The loss function to use in the model, default is 'mean_squared_error'
	# Returns:
	# 	model		(model)	: The model to be trained

	# Basic neural network
	model = Sequential()
	# Add layers to model based on the length of layer_size
	for layer, size in enumerate(layer_size):
		# Create cell based on cell type
		if layer == 0:
			# Add input layer that needs input shape defined
			model.add(InputLayer(input_shape=input_shape))
			model.add(cell(units=size, return_sequences=True))
			# For som eadvances explanation of return_sequences:
			# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
			# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
			# As explained there, for a stacked LSTM, you must set return_sequences=True 
			# when stacking LSTM layers so that the next LSTM layer has a 
			# three-dimensional sequence input. 
		elif layer == len(layer_size) - 1:
			# Add output layer that doesn't need a return sequence
			model.add(cell(units=size))
		else:
			# Add hidden layer that needs a return sequence as it's a stacked LSTM but no input shape
			model.add(cell(units=size, return_sequences=True))
		# Add dropout after each layer to prevent overfitting
		model.add(Dropout(dropout))
		# The Dropout layer randomly sets input units to 0 with a frequency of 
		# rate (= 0.2 above) at each step during training time, which helps 
		# prevent overfitting (one of the major problems of ML).
	# Add final dense layer to output prediction
	model.add(Dense(len(feature_columns), activation='linear'))
	# Compile model with given optimizer and loss function
	model.compile(optimizer=optimizer, loss=loss)
	# The optimizer and loss are two important parameters when building an 
	# ANN model. Choosing a different optimizer/loss can affect the prediction
	# quality significantly. You should try other settings to learn; e.g.
	# optimizer='rmsprop'/'sgd'/'adadelta'/...
	# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...
	
	return model

def train_model(x_train, y_train, hyperparameters, refresh=True, save=True, model_dir='model', checkpoint_dir='checkpoints', logs_dir='logs'):
	# Trains the model
	# Params:
	# 	x_train		(list)	: The x training data used to train the model
	# 	y_train		(list)	: The y training data used to train the model
	#	refresh 	(bool)	: Whether to retrain the model even if it exists, default is False
	#	save		(bool)	: Whether to save the model locally if it doesn't already exist, default is True
	#	model_dir	(str)	: Directory to store model, default is 'model'
	# Returns:
	# 	model		(model)	: The trained model
	
	# Creates model directory if it doesn't exist
	if not os.path.isdir(model_dir):
		os.mkdir(model_dir)

	# # Creates checkpoint directory if it doesn't exist
	# if not os.path.isdir(checkpoint_dir):
	# 	os.mkdir(checkpoint_dir)

	# # Creates logs directory if it doesn't exist
	# if not os.path.isdir(logs_dir):
	# 	os.mkdir(logs_dir)

	# Create model name based on hyperparameters
	model_name_parts = [
		str(x_train.shape[1:]).replace(' ', ''),			# Input shape
		hyperparameters['cell'].__name__,					# Cell type
		'-'.join(map(str, hyperparameters['layer_size'])),	# Layer sizes
		hyperparameters['dropout'],							# Dropout rate
		hyperparameters['optimizer'],						# Optimizer
		hyperparameters['loss']								# Loss function
	]
	# Join model name parts together
	model_name = 'model_' + '_'.join(map(str, model_name_parts))

	# Checks if model file with same hyperparameters exists
	if os.path.exists(os.path.join(model_dir, f'{model_name}.keras')) and not refresh:
		# If file exists and model shouldn't be updated, import as keras model object
		model = load_model(os.path.join(model_dir, f'{model_name}.keras'))
		return model

	# Create model based on hyperparameters
	model = create_model(
		x_train.shape[1:],
		hyperparameters['cell'],
		hyperparameters['layer_size'],
		hyperparameters['dropout'],
		hyperparameters['optimizer'],
		hyperparameters['loss']
	)

	# # Save model checkpoints
	# checkpointer = ModelCheckpoint(
	# 	os.path.join(
	# 		checkpoint_dir,
	# 		model_name + '.ckpt'
	# 	),
	# 	save_weights_only=True,
	# 	save_best_only=True,
	# 	verbose=1
	# )
	# Save model logs
	# tensorboard = TensorBoard(log_dir=os.path.join(logs_dir, model_name))

	# Train model
	model.fit(
		x_train,
		y_train,
		epochs=hyperparameters['eopchs'],
		batch_size=hyperparameters['batch_size'],
		# callbacks=[checkpointer, tensorboard],
		verbose=1
	)
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

	# Save model if saving is on
	if save:
		model.save(os.path.join(model_dir, f'{model_name}.keras'))

	return model

def predict(model, scaler, x_test, dates, predictions=1, feature_columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']):
	# Uses model and data to make prediction
	# Params:
	# 	model			(model)	: The model previously generated to actually be tested
	# 	scaler			(scaler): The scaler used to process the data so it can be de-normalised
	# 	x_test			(list)	: The x testing data
	# 	dates			(list)	: The dates to be used as the index for the predicted_df
	# 	predictions		(int)	: The amount of days ahead to predict, default is 1
	# 	feature_columns	(list)	: The list of features to be used as the columns for the predicted_df, default is everything grabbed from yahoo
	# Returns:
	# 	predicted_df	(df)	: The predictions based on the test data

	# Repeat the prediction for the number of days ahead
	for i in range(predictions):
		# Predict the price
		prediction = model.predict(x_test)
		
		# Add the prediction to the x_test so that it can be used to predict the next day
		x_test = np.concatenate((x_test, prediction[:, np.newaxis, :]), axis=1)
		# Remove the first value of x_test to keep the shape correct
		x_test = np.delete(x_test, 0, axis=1)
		
		# Inverse transform the prediction to get the actual price
		prediction = scaler.inverse_transform(prediction)

		# Add the prediction to the prices
		try:
			# Test if prices exists
			prices
		except NameError:
			# If prices doesn't exist, create it
			prices = prediction
		else:
			# Add the prediction to prices
			prices = np.vstack((prices, prediction))

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

	# Create dataframe of predictions
	predicted_df = pd.DataFrame(index=dates, columns=feature_columns, data=prices)

	return predicted_df

def candlestick(test_df, predicted_df, days=1, feature_columns=['Open', 'High', 'Low', 'Adj Close', 'Close', 'Volume']):
	# Uses model and data to make prediction
	# Params:
	# 	test_df			(df)	: The test data downloaded from yahoo
	# 	predicted_df	(df)	: The predictions based on the test data
	# 	days			(int)	: The amount of days to show in a candlestick, default is 1
	# 	feature_columns	(list)	: The list of features graphed from the prediction, default is everything grabbed from yahoo
	# Returns:
	# 	fig				(fig)	: The figure object to be shown

	# make sure that the passed feature_columns exist in the dataframe
	for column in feature_columns:
		assert column in df.columns, f'{column} does noxt exist in the dataframe.'

	# Ensure that days is valid
	assert days >= 0, 'days must be bigger than or equal to 1'

	# Tell how the pd resample should treat the different vales
	aggregation = {
		'Open':'first',
		'High':'max',
		'Low':'min',
		'Close':'last',
		'Adj Close':'last',
		'Volume':'sum'
	}
	# Resample both the test data and the prediction data based on 'days'
	resampled_test_df = test_df.resample(f'{days}d').agg(aggregation)
	resampled_predicted_df = predicted_df.resample(f'{days}d').agg(aggregation)

	# Put graphs together in figure
	fig = make_subplots(
		rows=2,
		cols=1,
		shared_xaxes=True,
		vertical_spacing=0.02,
		row_heights=[0.8, 0.2]
	)
	# Create candlestick graph of test data
	fig.add_trace(graphs.Candlestick(
		x=resampled_test_df.index,
		open=resampled_test_df['Open'],
		high=resampled_test_df['High'],
		low=resampled_test_df['Low'],
		close=resampled_test_df['Close'],
		showlegend=False
	), row=1, col=1)
	# Create scatter graphs of predicted feature
	for column in feature_columns:
		fig.add_trace(graphs.Scatter(
			x=resampled_predicted_df.index,
			y=resampled_predicted_df[column],
			name=f'Predicted {column}'
		), row=1, col=1)
	# Create volume graph of test data
	fig.add_trace(graphs.Bar(
		x=resampled_test_df.index,
		y=resampled_test_df['Volume'],
		name='Volume',
		showlegend=False
	), row=2, col=1)
	# Update figure titles
	fig.update_layout(
		title=f'{COMPANY} Share Prices {test_df.index[0]:%b %Y} - {test_df.index[-1]:%b %Y}',
		yaxis_title='Price ($)',
		xaxis_rangeslider_visible=False
	)

	return fig

def boxplot(test_df, predicted_df, days=1, feature_columns=['Open', 'High', 'Low', 'Adj Close', 'Close', 'Volume']):
	# Uses model and data to make prediction
	# Params:
	# 	test_df			(df)	: The test data downloaded from yahoo
	# 	predicted_df	(df)	: The predictions based on the test data
	# 	days			(int)	: The amount of days to show in a candlestick, default is 1
	# 	feature_columns	(list)	: The list of features graphed from the prediction, default is everything grabbed from yahoo
	# Returns:
	# 	fig				(fig)	: The figure object to be shown

	# make sure that the passed feature_columns exist in the dataframe
	for column in feature_columns:
		assert column in df.columns, f'{column} does noxt exist in the dataframe.'

	# Ensure that days is valid
	assert days >= 0, 'days must be bigger than or equal to 1'

	# Tell how the pd resample should treat the different vales
	aggregation = {
		'Open':'first',
		'High':'max',
		'Low':'min',
		'Close':'last',
		'Adj Close':'last',
		'Volume':'sum'
	}
	# Resample both the test data and the prediction data based on 'days'
	resampled_test_df = test_df.resample(f'{days}d').agg(aggregation)
	resampled_predicted_df = predicted_df.resample(f'{days}d').agg(aggregation)

	# Create a box plot for each day
	boxes = []
	for date in resampled_test_df.index:
		# Select the data for this day
		day_data = resampled_test_df[resampled_test_df.index == date][['Open', 'High', 'Low', 'Close', 'Adj Close']]
		# Create a box plot for this day's data
		box = graphs.Box(
			y=day_data.values[0],
			name=str(date),
			showlegend=False
		)
		boxes.append(box)
	# Create scatter graphs of predicted feature
	predicted_data_graphs = []
	for column in feature_columns:
		graph = graphs.Scatter(
			x=resampled_predicted_df.index,
			y=resampled_predicted_df[column],
			name=f'Predicted {column}'
		)
		predicted_data_graphs.append(graph)
	# Put graphs together in figure
	fig = graphs.Figure(data=(boxes + predicted_data_graphs))
	# Update figure titles
	fig.update_layout(
		title=f'{COMPANY} Share Prices {test_df.index[0]:%b %Y} - {test_df.index[-1]:%b %Y}',
		yaxis_title='Price ($)'
	)

	return fig

def error(test_df, predicted_df, feature_columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']):
	# Uses model and data to assess prediction
	# Params:
	# 	test_df			(df)	: The test data downloaded from yahoo
	# 	predicted_df	(df)	: The predictions based on the test data
	# 	feature_columns	(list)	: The list of features graphed from the prediction, default is everything grabbed from yahoo
	# Returns:
	# 	feature_error	(dict)	: The average error in dollars and percentage for each feature

	# Create dictionary to store the average error and average error percentage for each feature
	feature_error = {}
	for column in feature_columns:
		# Calculate the average error for each feature
		average_error = np.mean(np.abs(test_df[column] - predicted_df[column]))
		# Calculate the average error percentage for each feature
		average_error_percentage = average_error / np.mean(test_df[column])
		# Add the average error and average error percentage to the feature_error dict
		feature_error[column] = (average_error, average_error_percentage)

	return feature_error

if __name__ == '__main__':
	# Default params that must be set
	COMPANY = 'TSLA'
	START_DATE = '2015-01-01'
	END_DATE = '2023-09-29'
	SPLIT = 0.1
	CHOSEN_FEATURE = 'Close'
	REFRESH = False
	GRAPH_DAYS = 7
	FUTURE_DAYS = 7
	
	# Make model hyperparameters
	hyperparameters = {
		'sequence_length': 120,
		'cell': GRU,
		'layer_size': [50, 50, 50],
		'dropout': 0.2,
		'optimizer': 'adam',
		'loss': 'mean_squared_error',
		'eopchs': 25,
		'batch_size': 32
	}

	# Generate the dataset based on the company and the dates
	df = load_data(COMPANY, START_DATE, END_DATE, REFRESH)

	# Prepare the data for training and testing
	dataset = prepare_data(df, hyperparameters['sequence_length'], SPLIT)

	# Generate the model based on the training data
	model = train_model(dataset['x_train'], dataset['y_train'], hyperparameters, REFRESH)

	# Make df of predictions to compare against test data
	prediction_df = predict(model, dataset['scaler'], dataset['x_test'], dataset['test_df'].index)

	# Show candlestick graph of prices
	candlestick(dataset['test_df'], prediction_df, GRAPH_DAYS, [CHOSEN_FEATURE]).show()
	# Show boxplot of prices
	boxplot(dataset['test_df'], prediction_df, GRAPH_DAYS, [CHOSEN_FEATURE]).show()

	# Calculate the average error for each feature
	error = error(dataset['test_df'], prediction_df)
	for column in error:
		if column == 'Volume':
			print(f'Average error for {column}: {error[column][0]:.2f} or {error[column][1]:.2%}')
		else:
			print(f'Average error for {column}: ${error[column][0]:.2f} or {error[column][1]:.2%}')

	# Make datetime range for future predictions
	tomorrow = datetime.strptime(END_DATE, '%Y-%m-%d') + timedelta(days=1)
	dates = pd.date_range(tomorrow, periods=FUTURE_DAYS, name='Date')
	# Predict the future prices
	future_prices = predict(model, dataset['scaler'], dataset['model_inputs'], dates, FUTURE_DAYS)
	for day, value in enumerate(future_prices[CHOSEN_FEATURE]):
		print(f'Predicted {CHOSEN_FEATURE} price in {day + 1} day(s): ${value:.2f}')