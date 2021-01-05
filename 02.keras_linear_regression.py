import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

csv_dataset_file_path = r'E:\Amit_workspace\Projects\Projects_2021\Tensorflow-keras-scripts\Tensorflow-keras-scripts\dataset\stock_dataset\NSE-TATAGLOBAL.csv'
dataset_train = pd.read_csv(csv_dataset_file_path)


training_set = dataset_train.iloc[:, 1:2].values
print(dataset_train.head())
print(training_set)
for i in training_set:
    print(i)

# we have to scale our data for optimal performance. 
