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

def load_data(company, start_date, end_date, refresh=True, save=True, data_dir='data'):
	# Loads data from Yahoo Finance source.
    # Params:
	# 	company		(str)	: The company you want to train on, examples include AAPL, TESL, etc.
	#	start_date	(str)	: The start date for the data
	#	end_date	(str)	: The end date for the data
	#	refresh		(bool)	: Whether to redownload data even if it exists, default is True
	#	save		(bool)	: Whet	her to save the data locally if it doesn't already exist, default is True
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
	# 	df				(df)	: The dataframe to be processed
	# 	sequence_length	(int)	: The historical sequence length used to predict
	# 	split			(float)	: The percentage of data to be used for testing, default is 0.2
	# 	feature_columns	(list)	: The list of features to use to feed into the model, default is everything grabbed from yahoo

	# make sure that the passed feature_columns exist in the dataframe
	for col in feature_columns:
		assert col in df.columns, f'{col} does noxt exist in the dataframe.'

	try:
		split_is_date = isinstance(datetime.strptime(split, '%Y-%m-%d'), datetime)
	except:
		split_is_date = False
	split_is_float = isinstance(split, float) and 0 < split < 1
	
	# Ensure that split is valid
	assert split_is_date or split_is_float, f'split must be a float between 0 and 1 or a string (\'yyyy-mm-dd\')'

	# If split is false, split the dataframe into training and testing data based on the split
	if split_is_float:
		# Convert the split percentage to a index
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

	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled_data = scaler.fit_transform(df[feature_columns].values)

	x_train, y_train = [], []
	# Prepare training data
	for i in range(sequence_length, len(train_df)):
		# Add the previous sequence_length values to x_train
		x_train.append(scaled_data[i - sequence_length: i])
		# Add the current value to y_train
		y_train.append(scaled_data[i])

	x_test, y_test = [], []
	# Prepare testing data
	for i in range(len(train_df), len(scaled_data)):
		# Add the previous sequence_length values to x_test
		x_test.append(scaled_data[i - sequence_length: i])
		# Add the current value to y_test
		y_test.append(scaled_data[i])

	# Convert them into numpy arrays
	x_train, y_train = np.array(x_train), np.array(y_train)
	x_test, y_test = np.array(x_test), np.array(y_test)

	# Reshape the data to be 3-dimensional in the form [number of samples, sequence length, number of features]
	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], len(feature_columns)))
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], len(feature_columns)))

	model_inputs = x_test[-sequence_length:, 0]
	model_inputs = np.reshape(model_inputs, (1, model_inputs.shape[0], len(feature_columns)))

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
	for i in range(len(layer_size)):
		# Create cell based on cell type
		if i == 0:
			# Add input layer that needs input shape defined
			model.add(InputLayer(input_shape=input_shape))
			model.add(cell(units=layer_size[i], return_sequences=True))
			# For som eadvances explanation of return_sequences:
			# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
			# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
			# As explained there, for a stacked LSTM, you must set return_sequences=True 
			# when stacking LSTM layers so that the next LSTM layer has a 
			# three-dimensional sequence input. 
		elif i == len(layer_size) - 1:
			# Add output layer that doesn't need a return sequence
			model.add(cell(units=layer_size[i]))
		else:
			# Add hidden layer that needs a return sequence as it's a stacked LSTM but no input shape
			model.add(cell(units=layer_size[i], return_sequences=True))
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

def train_model(x_train, y_train, hyperparameters, refresh=True, save=True, model_dir='model'):
	# Trains the model
	# Params:
	# 	x_train		(list)	: The x training data
	# 	y_train		(list)	: The y training data
	#	refresh 	(bool)	: Whether to retrain the model even if it exists, default is False
	#	save		(bool)	: Whether to save the model locally if it doesn't already exist, default is True
	#	model_dir	(str)	: Directory to store model, default is 'model'
	# Returns:
	# 	model		(model)	: The trained model
	
	# Creates model directory if it doesn't exist
	if not os.path.isdir(model_dir):
		os.mkdir(model_dir)

	# Model filename to ensure model is unique
	model_file_path = os.path.join(model_dir, f"model_{x_train.shape[1:]}_{hyperparameters['cell'].__name__}_{'_'.join(map(str, hyperparameters['layer_size']))}_{hyperparameters['dropout']}_{hyperparameters['optimizer']}_{hyperparameters['loss']}.keras")

	# Checks if data file with same data exists
	if os.path.exists(model_file_path) and not refresh:
		# If file exists and data shouldn't be updated, import as pandas data frame object
		# 'index_col=0' makes the date the index rather than making a new coloumn
		model = load_model(model_file_path)
		return model

	# Create model based on hyperparameters
	model = create_model(x_train.shape[1:], hyperparameters['cell'], hyperparameters['layer_size'], hyperparameters['dropout'], hyperparameters['optimizer'], hyperparameters['loss'])

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

	if save:
		model.save(model_file_path)

	return model

def predict(model, scaler, x_test, dates, predictions=1, feature_columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']):
	# Uses model and data to make prediction
	# Params:
	# 	model				(model)	: The model previously generated to actually be tested
	# 	scaler				(scaler): The scaler used to process the data so it can be de-normalised
	# 	x_test				(list)	: The x testing data
	# 	dates				(list)	: The dates to be used for the predictions
	# 	predictions			(int)	: The amount of days to predict, default is 1
	# 	feature_columns		(list)	: The list of features to be predicted on, default is everything grabbed from yahoo
	# Returns:
	# 	predicted_df		(df)	: The predictions based on the test data

	for i in range(predictions):
		# Predict the price
		prediction = model.predict(x_test)
		# Add the prediction to the x_test
		x_test = np.concatenate((x_test, prediction[:, np.newaxis, :]), axis=1)
		# Remove the first value of x_test to keep the length the same
		x_test = np.delete(x_test, 0, axis=1)
		# Inverse transform the prediction to get the actual price
		prediction = scaler.inverse_transform(prediction)
		# # Add the prediction to the predicted_df
		# predicted_df.iloc[i] = prediction[0]

		# Add the prediction to the prices
		try:
			# If prices doesn't exist, create it
			prices
		except NameError:
			# Set prices to the prediction
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
	# 	feature_error	(dict)	: The average error and average error percentage for each feature

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
	END_DATE = '2023-09-27'
	SEQUENCE_LENGTH = 120
	CHOSEN_FEATURE = 'Close'
	REFRESH = False
	GRAPH_DAYS = 7
	FUTURE_DAYS = 7
	
	# Make model hyperparameters
	hyperparameters = {
		'cell': GRU,
		'layer_size': [50, 50, 50],
		'dropout': 0.2,
		'optimizer': 'adam',
		'loss': 'mean_squared_error'
	}

	# Generate the dataset based on the company and the dates
	df = load_data(COMPANY, START_DATE, END_DATE, REFRESH)

	# Prepare the data for training and testing
	dataset = prepare_data(df, SEQUENCE_LENGTH, 0.1)

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
		print(f'Average error for {column}: ${error[column][0]:.2f} or {error[column][1]:.2%}')

	# Make datime range for future predictions
	tomorrow = datetime.strptime(END_DATE, '%Y-%m-%d') + timedelta(days=1)
	dates = pd.date_range(tomorrow, periods=FUTURE_DAYS, name='Date')
	# Predict the future prices
	future_prices = predict(model, dataset['scaler'], dataset['model_inputs'], dates, FUTURE_DAYS)
	for day, value in enumerate(future_prices[CHOSEN_FEATURE]):
		print(f'Predicted {CHOSEN_FEATURE} price in {day + 1} day(s): ${value:.2f}')