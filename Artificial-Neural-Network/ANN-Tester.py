import tensorflow as tf
from skimage.feature import hog
from skimage.io import imread, imshow
import numpy as np
import cv2
import os

model = tf.keras.models.load_model("ANN-randomised-32x32-32-100.h5")



def tester():
	#do this
	dir_name = "../Fruit-Images-Dataset-master/Test"

	tested = 0
	correct = 0
	incorrect = 0

	fruits_list = {}

	count = 0
	for folder in os.listdir(dir_name):
		fruits_list[count] = folder
		count += 1		

	for folder in os.listdir(dir_name):

		folder_tested = 0
		folder_correct = 0

		for fruit in os.listdir(f"{dir_name}/{folder}"):

			image = imread(f"{dir_name}/{folder}/{fruit}")
			prediction_index = test(image)
			prediction = fruits_list[prediction_index]

			tested += 1
			folder_tested += 1

			if prediction == folder:
				correct += 1
				folder_correct += 1

			else:
				incorrect += 1

		print(f"Done with Folder of {folder}, {(folder_correct/folder_tested)*100}% accuracy.")
		print(f"{tested} tested so far with {(correct/tested)*100}% accuracy.")

def test(image):
	#does this
	resized = cv2.resize(image, (32, 32))
	vector = resized.flatten()
	x = vector.reshape(1, 3072)
	prediction = model.predict([x])


	maximum = 0
	max_index = ""
	out = []

	for i in range(len(list(prediction[0]))):
		
		if ( list(prediction[0])[i] > maximum ):
		
			maximum = list(prediction[0])[i]
			max_index = i

		#rounded probabilities added to out
		out.append(round(list(prediction[0])[i]))

	return max_index


tester()



