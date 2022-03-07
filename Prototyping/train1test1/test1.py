import pickle
import os
import numpy as np
from skimage.io import imread, imshow
from skimage.feature import hog
import matplotlib.pyplot as plt

with open("prototypes.pkl", "rb") as file:
	prototypes = pickle.load(file)
	
# test_img = imread("Fruit-Images-Dataset-master/Test/Cucumber Ripe/221_100.jpg")
# feature_vector = hog(test_img, orientations=11, pixels_per_cell=(5, 5), cells_per_block=(2, 2), multichannel=True)

def tester(feature_vector):
	min_distance = ""
	min_distance_img = ""

	for prototype in prototypes:

		for sub_type in prototypes[prototype]:


			dist = float(np.linalg.norm( feature_vector - sub_type ))

			if min_distance == "":
				min_distance = dist
				min_distance_img = prototype

			elif dist < min_distance:
				min_distance = dist
				min_distance_img = prototype

	return min_distance_img


def test():
	tested = 0
	correct = 0
	incorrect = 0

	dir_name = "Fruit-Images-Dataset-master/Test"

	for folder in os.listdir(dir_name):

		for fruit in os.listdir(dir_name + f"/{folder}"):

			img1 = imread(dir_name + f"/{folder}/{fruit}")

			vector = hog(img1, orientations=5, pixels_per_cell=(10, 10), cells_per_block=(4, 4), multichannel=True)

			tested += 1

			prediction = tester(vector)

			if prediction == folder:
				correct += 1

			else:
				incorrect += 1

			if tested % 50 == 0:
				print(f"{correct} out of {tested} with {(correct*100)/tested}% accuracy.")


	return tested, correct, incorrect


no_of_pics, match, unmatched = test()

print(no_of_pics)
print(match)
print(f"{match} out of {no_of_pics} with {(match*100)/no_of_pics}% accuracy.")
