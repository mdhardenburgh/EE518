import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def convertToOneHot(number):
    retVal = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    if number == 0:
        retVal = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    if number == 1:
        retVal = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    if number == 2:
        retVal = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    if number == 3:
        retVal = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    if number == 4:
        retVal = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    if number == 5:
        retVal = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    if number == 6:
        retVal = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    if number == 7:
        retVal = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    if number == 8:
        retVal = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    if number == 9:
        retVal = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    return retVal

data = loadmat('mnistFull.mat')
X_train = data['X_train'] /255.0 # training images
X_test = data['X_test'] /255.0 # test images
y_train = data['y_train'][:,0] # training labels
y_test = data['y_test'][:,0] # test labels
train_labels = []
train_images = []
test_labels = []
test_images = []
regularization_parameter = 1E-4

## YOUR CODE BELOW
# convert labels to one-hot encoding
for iter in y_train:
    train_labels.append(convertToOneHot(iter))
for iter in y_test:
    test_labels.append(convertToOneHot(iter))
# convert to array of array of one hot encodings
train_labels_array = np.array(train_labels, dtype=np.float16)
train_images = np.array(X_train, dtype=np.float16)
test_labels_array = np.array(test_labels, dtype=np.float16)
test_images = np.array(y_test, dtype=np.float16)
# solve for W
# train
I = np.eye(train_images.shape[1], dtype=np.float16)
W = (np.linalg.inv(train_images.T @ train_images + regularization_parameter * I))@train_images.T@train_labels

# report training and test error
# get training and test predictions
train_pred = train_images @ W
test_pred = X_test @ W

# Predict the labels for training and test datasets
train_pred_labels = np.argmax(train_pred, axis=1)
test_pred_labels = np.argmax(test_pred, axis=1)

# Calculate and print the error rates
train_error = np.mean(train_pred_labels != y_train)
test_error = np.mean(test_pred_labels != y_test)

print(f'Training Error: {train_error:.2f}')
print(f'Test Error: {test_error:.2f}')