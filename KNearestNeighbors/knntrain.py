import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from skimage.io import imread, imshow
from skimage.filters import prewitt_h,prewitt_v
from skimage.feature import hog


def trainer():
	fruit_vectors = {}
	prototype_vectors = {}
	fruit_vectors_a = {}
	flattened_images = {}


	dir_name = "Fruit-Images-Dataset-master/Training"
	count = 0


	for folder in os.listdir(dir_name):
		fruit_vectors[folder] = [[],[],[],[]]
		fruit_vectors_a[folder] = []
		flattened_images[folder] = []

		for image_file in os.listdir(dir_name + f"/{folder}"):

			img = imread(dir_name + f"/{folder}/{image_file}")
			feature_vector = hog(img, orientations = 4, pixels_per_cell = (20, 20), cells_per_block = (3, 3), multichannel = True)

			fruit_vectors_a[folder].append(feature_vector)

			arr = img.flatten()
			flattened_images[folder].append(arr)

			if image_file.split("_")[0] == "r":
				#Do this
				fruit_vectors[folder][0].append(feature_vector)

			elif image_file.split("_")[0] == "r2":
				#do this
				fruit_vectors[folder][1].append(feature_vector)

			elif image_file.split("_")[0] == "r3":
				#do this
				fruit_vectors[folder][2].append(feature_vector)

			else:
				# do this
				fruit_vectors[folder][3].append(feature_vector)

		

		# fruit_vectors[folder].append(np.mean(fruit_vectors[folder], axis = 0))
		prototype_vectors[folder] = [np.mean(fruit_vectors[folder][0], axis = 0), np.mean(fruit_vectors[folder][1], axis = 0), np.mean(fruit_vectors[folder][2], axis = 0), np.mean(fruit_vectors[folder][3], axis = 0)]
		
		count += 1
		
		print(f"Done with #{count} {folder}")

	# with open("fruits.pkl", "wb") as file1:
	# 	pickle.dump(fruit_vectors, file1)
	with open("prototypes.pkl","wb") as file2:
		pickle.dump(prototype_vectors,file2)
	with open("features.pkl","wb") as file3:
		pickle.dump(fruit_vectors_a,file3)
	with open("flattened.pkl","wb") as file4:
		pickle.dump(flattened_images,file4)


trainer()