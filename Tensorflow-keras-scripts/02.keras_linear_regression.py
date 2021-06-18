import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


csv_dataset_file_path = r'E:\Amit_workspace\Projects\Projects_2021\Tensorflow-keras-scripts\Tensorflow-keras-scripts\dataset\stock_dataset\NSE-TATAGLOBAL.csv'
dataset_train = pd.read_csv(csv_dataset_file_path)


training_set = dataset_train.iloc[:, 1:2].values
print(dataset_train.head())
print(training_set)
# for i in training_set:
#     print(i)

# we have to scale our data for optimal performance.
sc = MinMaxScaler(feature_range = (0, 1))
print(sc)
training_set_scaled = sc.fit_transform(training_set)
print(training_set_scaled)

# Creating Data with Timesteps

# LSTMs expect our data to be in a specific format, usually a 3D array.
# We start by creating data in 60 timesteps and converting it into an array using NumPy. 
# Next, we convert the data into a 3D dimension array with X_train samples, 60 timestamps, and one feature at each step.

# print(len(training_set))
n = len(training_set)
X_train = []
y_train = []
for i in range(60, n):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape) # (1975, 60, 1)


### Defining our LSTM model ###
'''We add the LSTM layer and later add a few Dropout layers to prevent overfitting. We add the LSTM layer with the following arguments:
50 units which is the dimensionality of the output space
return_sequences=True which determines whether to return the last output in the output sequence, or the full sequence
input_shape as the shape of our training set. 
When defining the Dropout layers, we specify 0.2, meaning that 20% of the layers will be dropped. 
Thereafter, we add the Dense layer that specifies the output of 1 unit. 
After this, we compile our model using the popular adam optimizer and set the loss as the mean_squarred_error. 
This will compute the mean of the squared errors. Next, we fit the model to run on 100 epochs with a batch size of 32. 
Keep in mind that, depending on the specs of your computer, this might take a few minutes to finish running.

'''
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)

regressor.save('Lstm_linear_regression.h5')