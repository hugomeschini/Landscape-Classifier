# Landscape Classifier 
### Python - Convolutional Neural Network
#### Create an automatic landscape classifier (images)

1. Upload the images. See how the data is stored. You will have to go through the folders, load the images into memory and label them with the names of the folders. Make a reshape of each image (start the exercise with 32x32, to go faster in the executions). You can use your own images or download the dataset considering the below link:
  - Train: https://drive.google.com/file/d/1HIL0wzQWKrcY8qmAFt5aJKaES_-aHLO7/view
  - Test: https://drive.google.com/file/d/166fEZRkJ4xaGT2aSyj1I6ph9q3QkVr8M/view
2. Investigate the images, check with some samples that you have loaded the data correctly.
3. Normalize
4. Design the architecture of the network. Remember that it is a classification algorithm. Be careful with the dimensions of the entrance.
5. Separate 20% of the training data to validate.
6. Represents the history object
7. Evaluate the model with the test data
8. Render some of the landscapes where the model makes mistakes
9. Create a confusion matrix with the model


## Libraries

- import numpy as np
- import pandas as pd 
- import matplotlib.pyplot as plt
- from keras.preprocessing.image import ImageDataGenerator, load_img
- from sklearn.model_selection import train_test_split
- from sklearn.utils import shuffle
- import tensorflow as tf
- import seaborn as sns
- from sklearn.metrics import confusion_matrix
- import matplotlib.pyplot as plt
- import random
- import os
- import cv2
