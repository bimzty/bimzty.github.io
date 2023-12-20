# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 22:33:23 2023

@author: 86182
"""
#newst insight:There are currently some limitations with using the vanilla LSTMs described above, specifically in the use of a financial time series, the series itself has non-stationary properties which is very hard to model (although advancements have been made in using Bayesian Deep Neural Network methods for tackling non-stationarity of time series). Also for some applications it has also been found that newer advancements in attention based mechanisms for neural networks have out-performed LSTMs (and LSTMs coupled with these attention based mechanisms have outperformed either on their own).
#Import Data
import pandas as pd

file_path = r'D:\Desktop\学习\我\申研综合研究\202302上证指数.csv'
data = pd.read_csv(file_path,encoding='GBK')

# Printing
print(data)
 
#Feature Enginnering
#%% mobile average
data['MA5'] = data['close'].rolling(window=5).mean()
data['MA10'] = data['close'].rolling(window=10).mean()

#%% Relative Strength Index，RSI）
window = 14
delta = data['close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
average_gain = gain.rolling(window=window).mean()
average_loss = loss.rolling(window=window).mean()
relative_strength = average_gain / average_loss
data['RSI'] = 100 - (100 / (1 + relative_strength))

#%% of Williams%R): The William indicators of the stock can be calculated to determine the situation of over -purchase or oversold. You can use the Williams_r () function in the TA library to calculate the William index
window = 14
highest_high = data['high'].rolling(window=window).max()
lowest_low = data['low'].rolling(window=window).min()
data['WilliamsR'] = (highest_high - data['close']) / (highest_high - lowest_low) * -100
 
#%% Bollinger Bands: It can calculate the Bollinger indicator of the stock to observe the upper and lower limits of price fluctuations. You can use the Bollinger Bands () function in the TA library to calculate the Blin belt index.
window = 20
std = data['close'].rolling(window=window).std()
data['BB_Middle'] = data['close'].rolling(window=window).mean()
data['BB_Upper'] = data['BB_Middle'] + 2 * std
data['BB_Lower'] = data['BB_Middle'] - 2 * std
#%%plot
# import numpy as np
import matplotlib.pyplot as plt

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# Plot Moving Average
ax1.plot(data['tdate'].to_numpy(), data['close'].to_numpy(), label='Close Price')
ax1.plot(data['tdate'].to_numpy(), data['MA5'].to_numpy(), label='MA5')
ax1.plot(data['tdate'].to_numpy(), data['MA10'].to_numpy(), label='MA10')
ax1.set_ylabel('Price')
ax1.set_title('Moving Average')
ax1.legend()

# Plot Relative Strength Index (RSI)
ax2.plot(data['tdate'].to_numpy(), data['RSI'].to_numpy())
ax2.set_xlabel('Date')
ax2.set_ylabel('RSI')
ax2.set_title('Relative Strength Index (RSI)')

# Adjust layout spacing
plt.tight_layout()

# Display the plot
plt.show()

#basic four feature constructed

 
#%% Increasing feature: financial market experience characteristics
import pandas as pd

# Calculate price change
data['price_change'] = data['close'] - data['open']

# Calculate daily range
data['range'] = data['high'] - data['low']

# Calculate volume change
data['volume_change'] = data['cjl'] - data['cjl'].shift(1)

# Calculate additional moving averages
data['MA20'] = data['close'].rolling(window=20).mean()
data['MA50'] = data['close'].rolling(window=50).mean()

# Calculate price-to-volume ratio
data['pvr'] = data['cje'] / data['cjl']

# Calculate high-low percentage change
data['hl_percentage_change'] = (data['high'] - data['low']) / data['low']

# Calculate price difference from moving averages
data['ma_diff'] = data['close'] - data['MA5']

# Display the updated dataset
print(data)
# Eight features added
 
#%% Increasing more features: Based on waveform -based feature extraction (for the whole time period)
# import pandas as pd
# Extract data
# waveform_data = data[['tdate', 'open', 'close', 'high', 'low', 'cjl', 'cje', 'hsl']]

# # Define waveform feature extraction function
# def extract_features(waveform):
#     waveform['open'] = pd.to_numeric(waveform['open'], errors='coerce')
#     waveform['close'] = pd.to_numeric(waveform['close'], errors='coerce')
#     waveform['high'] = pd.to_numeric(waveform['high'], errors='coerce')
#     waveform['low'] = pd.to_numeric(waveform['low'], errors='coerce')

#     features = {}
#     features['max_value'] = waveform.max()
#     features['min_value'] = waveform.min()
#     features['mean_value'] = waveform.mean()
#     features['amplitude'] = waveform.max() - waveform.min()
#     features['std_deviation'] = waveform.std()
#     features['volume_mean'] = waveform['cjl'].mean()
#     features['volume_max'] = waveform['cjl'].max()
#     features['volume_min'] = waveform['cjl'].min()
#     features['turnover_mean'] = waveform['cje'].mean()
#     features['turnover_max'] = waveform['cje'].max()
#     features['turnover_min'] = waveform['cje'].min()
#     features['turnover_rate_mean'] = waveform['hsl'].mean()
#     features['turnover_rate_max'] = waveform['hsl'].max()
#     features['turnover_rate_min'] = waveform['hsl'].min()
#     return features

# # Apply waveform feature extraction function
# extracted_features = extract_features(waveform_data)

# print features
# print(extracted_features)

#%% Basic data processing
#Feature Enginnering: data cleaning
import pandas as pd
import numpy as np

# read the data
#data = pd.read_csv('沪指3000-202302—withfeature.txt', delimiter='\t')

# replace 'name' to shindex'
data['name'] = 'shindex'

#replace the value
data['tdate'] = pd.to_datetime(data['tdate']).astype(np.int64)

# handle NAN and infinite
data = data.replace([np.inf, -np.inf], np.nan).fillna(0)

# New DataFrame
basic_process_data = pd.DataFrame(data)

# output
#basic_process_data.to_csv('basic_process_data.csv', index=False)
data = basic_process_data
#%%Feature Enginnering: Feature Transformation
#1.Distribution (skewness)
import pandas as pd
from scipy.stats import skew

# Filter numeric columns
numeric_cols = data.select_dtypes(include=[float, int]).columns

# Calculate skewness coefficient for each numeric column
skewness = data[numeric_cols].apply(lambda x: skew(x.dropna()))

# Print left-skewed and right-skewed values
print("Left-Skewed Values:")
print("------------------------")
left_skewed = skewness[skewness < -1]
if not left_skewed.empty:
    for col, skew_val in left_skewed.items():
        print(f"{col}\t\t{skew_val:.6f}")

print("\nRight-Skewed Values:")
print("------------------------")
right_skewed = skewness[skewness > 1]
if not right_skewed.empty:
    for col, skew_val in right_skewed.items():
        print(f"{col}\t\t{skew_val:.6f}")
#The partial state coefficient is positive, then the right partial distribution; the partial state coefficient is negative, then..

#Transfromation
import pandas as pd
import numpy as np
from scipy.stats import skew

# Filter numeric columns
numeric_cols = data.select_dtypes(include=[float, int]).columns

# Calculate skewness coefficient for each numeric column
skewness = data[numeric_cols].apply(lambda x: skew(x.dropna()))

# Apply appropriate transformation to left-skewed and right-skewed data
for col, skew_val in skewness.items():
    if skew_val < -1:  # Left-skewed
        data[col] = np.log1p(data[col])
    elif skew_val > 1:  # Right-skewed
        data[col] = np.sqrt(data[col])

# Print the transformed data
print("Transformed Data:")
print("------------------------")
print(data)
#%%Feature Enginnering: Exterme value
import pandas as pd
import numpy as np

# Select the numerical column for processing
numeric_columns = data.select_dtypes(include=np.number).columns

n = 3

# Calculate the mean and standard deviation of each column
means = data[numeric_columns].mean()
stds = data[numeric_columns].std()

for column in numeric_columns:
#Accut the upper and lower truncation thresholds
    upper_threshold = means[column] + n * stds[column]
    lower_threshold = means[column] - n * stds[column]

# Replace the value of exceeding the threshold to threshold
    data[column] = np.where(data[column] > upper_threshold, upper_threshold, data[column])
    data[column] = np.where(data[column] < lower_threshold, lower_threshold, data[column])

# Printing processing data
print(data)
#%%Feature Enginnering: Standardization
#Cross-sectional Standardization is a score that standardizes the observation value of data at different time points at different time points, so that the position of each observation value in all observations is relatively stable.
import pandas as pd
import numpy as np

numeric_columns = ['open', 'close', 'high', 'low', 'cjl', 'cje', 'hsl', 'MA5', 'MA10',
                   'RSI', 'WilliamsR', 'BB_Middle', 'BB_Upper', 'BB_Lower',
                   'price_change', 'range', 'volume_change', 'MA20', 'MA50',
                   'pvr', 'hl_percentage_change', 'ma_diff']

# Calculate the average and standard deviation of each column
means = data[numeric_columns].mean()
stds = data[numeric_columns].std()

# Standardize each observation value
for column in numeric_columns:
    data[column] = (data[column] - means[column]) / stds[column]

# print
print(data)
#%%Feature Enginnering: Fill Nan Value

#%%Model Construction
#GRU+NN stucture
split_ratio = 0.8  # Split data: 80%Train, 20%Test
split_index = int(len(data) * split_ratio)

train_data = data[:split_index]
test_data = data[split_index:]

#only use 'close' as target
#Mode1.GRU+NN
#GRU here can also be changed to LSTM
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import matplotlib.pyplot as plt

feature_columns = ['close']

features = train_data[feature_columns].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

target = test_data['close'].values

scaler_target = MinMaxScaler(feature_range=(0, 1))
scaled_target = scaler_target.fit_transform(target.reshape(-1, 1))

#Build the Neural Network
model = Sequential()
model.add(GRU(units=64, input_shape=(time_steps, 1))) # Shape (time_steps, 1)
model.add(Dense(units=32))
model.add(Dense(units=1))

#Compile the model
model.compile(optimizer='adam', loss='mse')

#Translate data into Window sequence
def create_sequence_data(X, y, time_steps):
    X_sequence, y_sequence = [], []
    for i in range(len(X) - time_steps):
        if i + time_steps < len(y): 
            # check the length
            X_sequence.append(X[i:i+time_steps].reshape(-1, time_steps, 1)) # 调整数据形状为 (time_steps, 1)
            y_sequence.append(y[i+time_steps])
    return np.array(X_sequence), np.array(y_sequence)

time_steps = 10 # Set the time window
X_sequence, y_sequence = create_sequence_data(scaled_features, scaled_target, time_steps)

#Train the model
model.fit(X_sequence, y_sequence, epochs=10, batch_size=32)

test_features = test_data[feature_columns].values
scaled_test_features = scaler.transform(test_features)
X_test_sequence, y_test_sequence = create_sequence_data(scaled_test_features, np.zeros(len(scaled_test_features)), time_steps)
predictions = model.predict(X_test_sequence)

scaled_predictions = scaler_target.inverse_transform(predictions)

#Plot closing prices of forecast results against test data
#Draw a comparison chart of the closing price of the prediction results and the test data
actual_close = test_data['close'].values

plt.plot(actual_close, label='Actual')
plt.plot(scaled_predictions, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Close Price')
plt.legend()
plt.show()

#Next, Reference article: 2.2 orthogonal factor performance: deep learning high -frequency factor and part of the low -frequency factor and logical -based high -frequency factor correlation.
#%% Use Self-Attention mechanism to improve the model
#phenomena: Lower frequency of train data result in a even better result

# From the article Haitong 79: Once the input sequence is too long, even the GRU and LSTM models, which are themselves used to solve the problem of "memorability" of the sequence, will "forget" a lot of information in the early data.
#Once the input sequence is too long, even the GRU and LSTM models, which are designed to solve the "memorability" problem of the sequence, will "forget" a lot of information in the earlier data.
# Here we choose Self-Attention mechanism. That is, a query-key-value pattern is used to calculate attention scores. by

#Use LSTM here, can also be changed to GRU
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
#Oouput a null array, denote no GPU available
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Attention
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam

#Construct self-attention model
inputs = Input(shape=(time_steps, len(feature_columns)))
lstm_output = LSTM(50, return_sequences=True)(inputs)
attention_output = Attention()([lstm_output, lstm_output])
output = Dense(len(feature_columns))(attention_output)

model = Model(inputs=inputs, outputs=output)

#Compile and train the model
model.compile(optimizer=Adam(), loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Make predictions on the test set

predictions = model.predict(X_test)

# Reverse-normalize the predicted results
scaled_predictions = scaler.inverse_transform(predictions)

#Plot closing prices of forecast results against test data
#Draw a comparison chart of the closing price of the prediction results and the test data
actual_close = test_data['close'].values

plt.plot(actual_close, label='Actual')
plt.plot(scaled_predictions, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Close Price')
plt.legend()
plt.show()
# Plot the predicted results and actual values on a chart

plt.plot(scaled_predictions, label='Predictions')

plt.plot(scaler.inverse_transform(y_test), label='Actual')
#%%Feature Enginnering by Feature attribution:
# Feature masking models use various techniques and metrics to quantify the importance of features, such as Saliency, etc. Common feature homing methods include gradient and gradient class methods (e.g., gradient x input, gradient x class, Integrated Gradients, etc.), activation values (e.g., activation maximization, DeepLIFT, etc.), and backpropagation based methods (e.g., Grad-CAM, Guided Backpropagation, etc.)
#Saliency method: (Saliency value is the feature weight after model fitting) [From Linear Regression]

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

time_window = 7

features = data.drop(['code', 'name', 'ktype', 'tdate'], axis=1).shift(1).rolling(time_window).mean()
target = data['price_change']

features = features.dropna()
target = target[features.index]

# Training model

# The linear model is used here, is it necessary to consider using other models

model = LinearRegression()

#model = DecisionTreeRegressor()

model.fit(features, target)



# Calculate the Saliency value of the feature

saliency = abs(model.coef_)



# Print Saliency values for each feature

for feature, value in zip(features.columns, saliency):

print(f'{feature}: {value}')



#Saliency bar chart:

# Draw a bar chart

plt.figure(figsize=(10, 6))

plt.bar(features.columns, saliency)

plt.xlabel('Features')

plt.ylabel('Saliency')

plt.title('Saliency Values of Features')

plt.xticks(rotation=45)

plt.show()
#%%2 IG Integrated Gradient method
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Read data file
# data = pd.read_csv('沪指3000-202302—withfeature.txt', delimiter='\t')
time_window = 7

# Extract features and target variable
features = data.drop(['code', 'name', 'ktype', 'tdate'], axis=1).shift(1).rolling(time_window).mean()
target = data['price_change']

# Remove rows containing NaN values
features = features.dropna()
target = target[features.index]

# Define and train the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(features.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse')
model.fit(features, target, epochs=10)

# Define a new function for computing IG using TensorFlow operations
@tf.function
def integrated_gradients(inputs):
    # Compute the baseline output
    baseline_output = model(inputs, training=False)
    
    # Compute the integrated gradients
    inputs_shape = tf.shape(inputs)
    inputs_flat = tf.reshape(inputs, [-1])
    baseline_flat = tf.reshape(baseline_output, [-1])
    
    alpha_values = tf.linspace(tf.cast(0, tf.float32), tf.cast(1, tf.float32), num=100)
    
    def interpolate_inputs(alpha):
        return baseline_flat + (inputs_flat - baseline_flat) * alpha
    
    interpolated_inputs = tf.vectorized_map(interpolate_inputs, alpha_values)
    interpolated_inputs = tf.reshape(interpolated_inputs, [-1] + inputs_shape[1:])
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        predictions = model(interpolated_inputs, training=False)
    
    grads = tape.gradient(predictions, interpolated_inputs)
    avg_grads = tf.reduce_mean(grads, axis=0)
    integrated_gradients = (inputs_flat - baseline_flat) * avg_grads
    integrated_gradients = tf.reshape(integrated_gradients, inputs_shape)
    
    return integrated_gradients

# Select a sample and compute feature importance
sample_index = 0  # Select the index of the sample to compute feature importance
sample_features = features.iloc[sample_index].values.reshape(1, -1)

# Compute feature importance using Integrated Gradients
feature_importance = integrated_gradients(sample_features)

# Compute the absolute importance of each feature
feature_importance_abs = np.abs(feature_importance)

# Print the absolute importance of each feature
for feature, importance in zip(features.columns, feature_importance_abs[0]):
    print(f'{feature}: {importance}')

import matplotlib.pyplot as plt

# Create a bar plot of feature importance
plt.figure(figsize=(10, 6))
plt.bar(features.columns, feature_importance_abs[0])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance (Integrated Gradients)')
plt.tight_layout()
plt.show()

# The following is the basic principle of IG method:
# Select a reference input feature: First, we need to select a reference input feature as a reference point. Typically, we choose a reference input feature that is similar to the real input feature but easy to calculate the gradient. For example, for image data, an all-black image can be selected as the reference input feature.
# Interpolation path: When calculating the importance of a feature, the IG method gradually changes the value of the input feature by interpolating on the path between the reference input feature and the real input feature. This path can consist of a series of steps or interpolation points.
#Prediction and gradient calculation: For each interpolation point, we use a deep learning model to predict it and calculate the gradient of the predicted result relative to the input features. The gradient represents how sensitive the model is to input features.
# Integral calculation: Calculate the integral between the predicted gradient of each interpolation point along the interpolation path and the reference input feature. This integral can be estimated by a variety of methods, such as using numerical integration methods (such as discrete sums of Riemann and gradients) or approximate integration methods (such as gradient averaging based on linear interpolation).
# Feature importance: Multiply the difference between the integral result and the input features to get an importance score for each feature. This importance score indicates how much each feature contributes to the model's predictions. A larger importance score indicates that the feature plays a more important role in the model prediction.
#%%3 From features importance in DecisionTree 
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# import data
#data = pd.read_csv('沪指3000-202302—withfeature.txt', delimiter='\t')
time_window = 7

# extarct features and values 
features = data.drop(['code', 'name', 'ktype', 'tdate', 'price_change'], axis=1).shift(1).rolling(time_window).mean()
target = data['price_change']

# Remove the row containing the nan value
features = features.dropna()
target = target[features.index]

# load the model
model = DecisionTreeRegressor()
modelDT = model.fit(features, target)

# calculate feature importance
sample_index = 0   
sample_features = features.iloc[sample_index].values.reshape(1, -1)

#calculate importance and print
feature_importance = model.feature_importances_

for feature, importance in zip(features.columns, feature_importance):
    print(f'{feature}: {importance}')

import matplotlib.pyplot as plt

# Get the predicted values from the model
predicted_values = model.predict(features)

# Calculate the baseline output (e.g., average or a specific value)
baseline_output = target.mean()  # Change this according to your baseline calculation

# Create a line plot to compare model output and baseline output
plt.figure(figsize=(10, 6))
plt.plot(predicted_values, label='Model Output')
plt.axhline(y=baseline_output, color='r', linestyle='--', label='Baseline Output')
plt.xlabel('Sample Index')
plt.ylabel('Output')
plt.title('Comparison of Model Output and Baseline Output')
plt.legend()
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Create a bar plot of feature importance
plt.figure(figsize=(10, 6))
plt.bar(features.columns, feature_importance)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

#Feature ranking based on the feature importance
feature_importance = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importance})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

feature_importance_df.plot(x='Feature', y='Importance', kind='bar')
#%%4. Shapely Value Sampling(Still have error)
import pandas as pd
import numpy as np

features = features.dropna()
target =  target.loc[features.index]
print(features)

import numpy as np
import shap
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define and train the model
model = Sequential()
model.add(Dense(2, activation='relu', input_shape=(features.shape[1],)))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse')
model.fit(features, target, epochs=10)

# Create Shapley value explainer
explainer = shap.KernelExplainer(model.predict, features)

# Calculate Shapley values
shap_values = explainer.shap_values(features)

# Plot Shapley values
shap.summary_plot(shap_values, features)
#%% Model GRU+NN with dropout and feature selection  
from tensorflow.keras.layers import Dropout
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np

# select the column for training
feature_columns = ['open', 'close', 'high', 'low', 'cjl', 'cje', 'hsl', 'MA5', 'MA10',
                   'RSI', 'WilliamsR', 'BB_Middle', 'BB_Upper', 'BB_Lower',
                   'price_change', 'range', 'volume_change', 'MA20', 'MA50',
                   'pvr', 'hl_percentage_change', 'ma_diff']
#if direct do that, have error Input X contains NaN.
features = data[feature_columns].values
imputer = SimpleImputer(strategy = 'mean')
features  = imputer.fit_transform(features)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

X = []
y = []

# Define the time step
time_steps = 20

# build the training set
for i in range(time_steps, len(scaled_features)):
    X.append(scaled_features[i - time_steps:i])
    y.append(scaled_features[i, feature_columns.index('close')])  # Use 'close' column as the target variable

# turn the data to numpy
X = np.array(X)
y = np.array(y)

frequency = 1 
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
X_train = X_train[::frequency]
y_train, y_test = y[:train_size], y[train_size:]
y_train = y_train[::frequency]

#for testing the code, not real ranking
selected_feature_indices=[8,9]
X_train = X_train[:, :, selected_feature_indices]
X_test = X_test[:, :, selected_feature_indices]
# Apply feature ranking to select the top k features
k = 2  # Number of top features to select
selector = SelectKBest(score_func=f_regression, k=k)
#The f_regression method is a statistical test that measures the linear relationship between each feature and the target variable. It calculates the F-value and p-value for each feature, where the F-value represents the strength of the relationship and the p-value represents the significance of the relationship.
selected_features = selector.fit_transform(X_train.reshape(X_train.shape[0], -1), y_train)
selected_feature_indices = selector.get_support(indices=True)
#error with original selection
#for testing the code, not real ranking
selected_feature_indices=[8,9]
# Update the feature columns based on the selected features
feature_columns = [feature_columns[i] for i in selected_feature_indices]
#%%run the model
# Extract the selected features from the scaled features
scaled_features = scaled_features[:, selected_feature_indices]

# Construct the model with dropout layers
model = Sequential()
model.add(GRU(50, input_shape=(time_steps, len(feature_columns))))
model.add(Dropout(0.2))  # Add a dropout layer with dropout rate 0.2
model.add(Dense(len(feature_columns)))

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict on the test set
predictions = model.predict(X_test)

# Inverse transform the predictions
#scaled_predictions = scaler.inverse_transform(predictions)

# Print the scaled predictions
print(predictions)
#%%Feature Engineering: Feature selection
#3.1 Static feature selection
#3.1.1 Select+DecisionTree
time_window = 7

features = data.drop(['code', 'name', 'ktype', 'tdate', 'price_change'], axis=1).shift(1).rolling(time_window).mean()

#The feature list with importance greater than 0.04
selected_feature = feature_importance_df.loc[feature_importance_df['Importance'] > 0.04, 'Feature'].tolist()
feature_selected = data[selected_feature]  

feature_selected = feature_selected.dropna()
target = data['price_change']
target = target[feature_selected.index]

model = DecisionTreeRegressor()
modelDT_WS = model.fit(feature_selected, target)
#%%3.2 Dynamic feature selection(报错：IndexError: positional indexers are out-of-bounds)
from sklearn.tree import DecisionTreeRegressor

# Set the time window
time_window = 7

# Check if the DataFrame has enough rows
if len(data) >= time_window:
    # Initialize an empty list to store selected features
    feature_dynamic_selected = []

    # Iterate over the range of time window movement
    for i in range(len(data) - time_window + 1):
        # Construct rolling mean features based on the current time window
        features = data.drop(['code', 'name', 'ktype', 'tdate', 'price_change'], axis=1).shift(1).rolling(time_window).mean()

        # Get the list of selected features with importance greater than 0.04 in the current time window
        selected_feature = feature_importance_df.loc[feature_importance_df['Importance'] > 0.04, 'Feature'].tolist()

        # Select the features from the current time window
        if i + time_window <= len(features):
            feature_selected = features.iloc[i:i+time_window][selected_feature]

            # Drop samples with missing values
            feature_selected = feature_selected.dropna()

            # Add the selected features from the current time window to the dynamic feature selection list
            feature_dynamic_selected.append(feature_selected)

    # Check if any features were selected
    if feature_dynamic_selected:
        # Concatenate all the selected features from different time windows
        feature_selected = pd.concat(feature_dynamic_selected)

        # Reset the index
        feature_selected.reset_index(drop=True, inplace=True)

        # Extract the target variable from the original data
        target = data['price_change']

        # Filter the target variable based on the feature selection results
        target = target.iloc[feature_selected.index]

        # Use a decision tree regressor model for modeling
        model = DecisionTreeRegressor()
        modelDT_WDS = model.fit(feature_selected, target)
    else:
        print("No features were selected.")
else:
    print("Not enough data points for the specified time window.")

#%%Explore: Find features from Neural Nework (using TSNE)
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Get the weights and biases of the output layer
output_layer = model.layers[-1]
weights = output_layer.get_weights()[0]
biases = output_layer.get_weights()[1]

# Print the weights and biases
print("Output layer weights:")
print(weights)
print("Output layer biases:")
print(biases)

#######################################

# Extract the hidden layer state
hidden_layer = model.layers[0]  # Assuming the hidden layer is the first layer
get_hidden_state = K.function([model.input], [hidden_layer.output])
hidden_state = get_hidden_state([X_train])[0]  # input_data is the input data

# Print the hidden layer state
print("Hidden layer state:")
print(hidden_state)

#######################################

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2)
hidden_state_tsne = tsne.fit_transform(hidden_state)

# Plot the scatter plot for visualization
plt.scatter(hidden_state_tsne[:, 0], hidden_state_tsne[:, 1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Hidden Layer State Visualization')
plt.show()

#######################################

# Extract the output of a convolutional layer
layer_index = 0  # Assuming the convolutional layer index is 0
conv_layer = model.layers[layer_index]
get_conv_output = K.function([model.input], [conv_layer.output])
conv_output = get_conv_output([X_train])[0]  # input_data is the input data

num_filters = 50

# Plot the filters/activation maps
plt.figure(figsize=(10, 6))
for i in range(num_filters):
    plt.subplot(4, 8, i+1)
    plt.imshow(conv_output[0, :, :, i], cmap='gray')
    plt.axis('off')
plt.suptitle('Convolutional Layer Filters/Activation Maps')
plt.show()

 
#%%Other ML mehtod:Linear regression
import matplotlib.pyplot as plt
import numpy as np

target_columns = ['close', 'high', 'low', 'cjl', 'cje', 'hsl']

fig, axs = plt.subplots(len(target_columns), 1, figsize=(8, 6 * len(target_columns)))

for i, column in enumerate(target_columns):
    axs[i].scatter(y_test[column], predictions[:, i], color='blue', label='Predicted')
    axs[i].scatter(y_test[column], y_test[column], color='red', label='Actual')
    axs[i].set_xlabel('Actual {}'.format(column))
    axs[i].set_ylabel('Predicted {}'.format(column))
    axs[i].set_title('Linear Regression - Actual vs Predicted - {}'.format(column))
    axs[i].legend()

# 调整子图布局
plt.tight_layout()

# 显示图表
plt.show()

#%% Decision Tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

# Build model
model = DecisionTreeRegressor()

# Fit model
model.fit(X_train, y_train)

# predict
predictions = model.predict(X_test)

# evaluation
mse = mean_squared_error(y_test, predictions)
print(mse)

# print the process
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True)
plt.show()

#%%RF
from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest regressor
modelRF = RandomForestRegressor()
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
# Train the model
modelRF.fit(X_train, y_train)

# Generate predictions
predictions = modelRF.predict(X_test)

# Scale the predictions
scaled_predictions = scaler.fit_transform(predictions.reshape(-1, 1)).flatten()

# Define buy and sell thresholds
buy_threshold = 0.02
sell_threshold = -0.02

# Initialize trading signals and position status
signals = np.zeros(len(scaled_predictions))
position = 0

# Generate trading signals (1 for buy, -1 for sell, 0 for hold)
for i in range(7, len(scaled_predictions)):
    window = scaled_predictions[i-7:i]
    if np.all(window > buy_threshold) and position != 1:
        position = 1
    elif np.all(window < sell_threshold) and position != -1:
        position = -1
    else:
        position = 0

    signals[i] = position

# Continue with the rest of the code for calculating statistics, etc.

#%%Calculate the revenue
# Initialize other variables and lists
weekly_returns = []
weekly_IC = []
week_predictions = []
week_actuals = []

# Iterate over the signals array
for i in range(7, len(signals)-1, 5):
    week_range = range(i-6, i+1)  # Get the range for the current week
    week_signals = signals[week_range]

    if np.count_nonzero(week_signals) > 0:  # Only consider weeks with non-zero signals
        week_predictions.append(modelDT_WS.predict(feature_selected[week_range]))

        # Flatten y_test to handle the appropriate dimensions
        flattened_y_test = np.squeeze(scaler.inverse_transform(y_test))
        week_actuals.append(flattened_y_test[week_range])

# Convert the lists to numpy arrays
week_predictions = np.array(week_predictions)
week_actuals = np.array(week_actuals)

# Calculate win rate for the weeks
win_rate = np.mean(week_predictions > 0, axis=0)

# Apply trading strategy to calculate weekly returns
weekly_return = (week_predictions[:, -1] - week_predictions[:, 0]) / week_predictions[:, 0]

# Calculate IC for the weeks
IC = np.corrcoef(week_predictions.flatten(), week_actuals.flatten())[0, 1]

# Append the last week's statistics if it was not a complete week
if len(signals) % 5 != 0:
    last_week_range = range(len(signals) - (len(signals) % 5), len(signals))
    last_week_signals = signals[last_week_range]

    if np.count_nonzero(last_week_signals) > 0:
        last_week_predictions = modelDT_WS.predict(feature_selected[last_week_range])
        last_week_actuals = flattened_y_test[last_week_range]

        week_predictions = np.append(week_predictions, [last_week_predictions], axis=0)
        week_actuals = np.append(week_actuals, [last_week_actuals], axis=0)

        win_rate = np.append(win_rate, np.mean(last_week_predictions > 0))
        weekly_return = np.append(weekly_return, (last_week_predictions[-1] - last_week_predictions[0]) / last_week_predictions[0])
        IC = np.corrcoef(np.append(week_predictions.flatten(), last_week_predictions.flatten()), np.append(week_actuals.flatten(), last_week_actuals.flatten()))[0, 1]

# Calculate average IC for the weeks
avg_IC = np.mean(IC)

# Print the results
print("Weekly Win Rate:", win_rate)
print("Weekly Returns:", weekly_return)
print("Average Weekly IC:", avg_IC)










