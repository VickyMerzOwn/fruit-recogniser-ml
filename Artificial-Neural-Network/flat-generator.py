import numpy as np
import pandas as pd
import pickle
import cv2
import os
from skimage.io import imread, imshow

def trainer():
	fruit_vectors = {}
	prototype_vectors = {}
	fruit_vectors_a = {}
	flattened_images = {}


	dir_name = "../Fruit-Images-Dataset-master/Training"
	count = 0


	for folder in os.listdir(dir_name):
		flattened_images[folder] = []

		for image_file in os.listdir(dir_name + f"/{folder}"):

			img = imread(dir_name + f"/{folder}/{image_file}")
			resized = cv2.resize(img, (32,32))

			arr = resized.flatten()
			flattened_images[folder].append(arr)
		
		count += 1
		print(f"Done with #{count} {folder}")

	flat_list = []

	for fruit in flattened_images:
		for image in flattened_images[fruit]:
			imlist = list(image)
			imlist.append(fruit)
			flat_list.append(imlist)

		print(f"Done with {fruit}")




	
	with open("resized-flattened.pkl1","wb") as file4:
		pickle.dump(flat_list, file4)


trainer()