import pickle
import os
from skimage.io import imread, imshow
from skimage.feature import hog
from numpy.linalg import norm

with open("features.pkl", "rb") as file:
    fruit_vectors = pickle.load(file)

# app = imread("apple.jpg")
# orange = imread("orange.jpg")

# app1 = cv2.resize(app, (100, 100))
# or1 = cv2.resize(orange, (100, 100))

# appvector = hog(app1, orientations=4, pixels_per_cell=(20, 20), cells_per_block=(3, 3), multichannel=True)
# orvector = hog(or1, orientations=4, pixels_per_cell=(20, 20), cells_per_block=(3, 3), multichannel=True)





# print(len(fruit_vectors['Apple Braeburn'][3][0]))
# img1 = imread("Fruit-Images-Dataset-master/Training/Apple Braeburn/0_100.jpg")
# vector = hog(img1, orientations=4, pixels_per_cell=(20, 20), cells_per_block=(3, 3), multichannel=True)
# print(len(vector))

#print(fruit_vectors.keys())
#print(len(fruit_vectors["Avocado"]))
#print(fruit_vectors["Avocado"])

def main():

# n = int(input("Enter the number of neighbours to see: ")

    tested = 0
    correct = 0
    incorrect = 0

    dir_name = "Fruit-Images-Dataset-master/Test"

    for folder in os.listdir(dir_name):

        for fruit in os.listdir(dir_name + f"/{folder}"):

            tested += 1
            img1 = imread(dir_name + f"/{folder}/{fruit}")
            # Now we have a feature vector of the test image

            vector = hog(img1, orientations=4, pixels_per_cell=(20, 20), cells_per_block=(3, 3), multichannel=True)
            
            dist = [] # This will tell me the distances
            fruits = {}
            
            for test_fruit in fruit_vectors:
                fruits[test_fruit] = 0
            
                #print(f'*************** DISTANCES FOR {test_fruit} ******************')
            
                for test_fruit_vector in fruit_vectors[test_fruit]:
            
                   distance = norm(vector - test_fruit_vector)
                   dist.append((distance,test_fruit))
            
            dist.sort(key=lambda x: x[0])
            
            for i in range(0,2):
            
                fruits[dist[i][1]] += 1
            
            prediction = ""
            mx = 0
            
            for y in fruits:
            
                if fruits[y] > mx:
            
                    mx = fruits[y]
                    prediction = y
            
            if prediction == folder:
                correct += 1
            else:
                incorrect += 1

            print(tested)

        print(f"The accuracy so far is {(correct/tested) * 100}%")
    
    print(correct/tested)

main()
