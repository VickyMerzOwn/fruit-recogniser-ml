import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import random


with open("resized-flattened.pkl1","rb") as file1:
	dfX = pickle.load(file1)



count = 0
dir_name = "../Fruit-Images-Dataset-master/Test"
fruit_types = {}

for folder in os.listdir(dir_name):
	count += 1
	fruit_types[folder] = count

random.shuffle(dfX)

dfX1 = pd.DataFrame(dfX)

X_train = dfX1.iloc[:, :-1].values
Y_train = dfX1.iloc[:, -1].values

y_train = []


for fruit in list(Y_train):
	vector = np.zeros(131, dtype = int)
	key = fruit_types[fruit] - 1
	vector[key] = 1
	y_train.append(vector)


y_train = np.array(y_train)


ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=300,activation="relu"))
ann.add(tf.keras.layers.Dense(units=256,activation="relu"))
# ann.add(tf.keras.layers.Dense(units=200,activation="relu"))
ann.add(tf.keras.layers.Dense(units=131,activation="softmax"))

ann.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])

ann.fit(X_train, y_train, batch_size=32, epochs = 100)

ann.save("ANN-randomised-32x32-32-100.h5")

