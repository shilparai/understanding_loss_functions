from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import cv2

def sigmoid_activation(x):
    return  1/(1+np.exp(x)) # compute the sigmoid activation value for input value x

def prediction(X,W):
    preds = sigmoid_activation(X.dot(W))
    preds[preds<=0.5] = 0
    preds[preds>0.5] = 0
    return preds

# we will be working on dog-cat data
# define the function to get the list of files in folders
def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

imagePaths = getListOfFiles("./pyimagesearch/datasets_mask/")

data = []
lables = []
c = 0
for image in imagePaths:

    lable = os.path.split(os.path.split(image)[0])[1]
    lables.append(lable)

    img = cv2.imread(image)
    img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA)
    data.append(img)
    c=c+1
    #print(c)

#print(lables)

# encode the labels as integer
data = np.array(data)
lables = np.array(lables)

le = LabelEncoder()
lables = le.fit_transform(lables)

myset = set(lables)
print(myset)

#dataset_size = data.shape[0]
#data = data.reshape(dataset_size,-1)

print(data.shape)
print(lables.shape)
#print(dataset_size)

(trainX, testX, trainY, testY ) = train_test_split(data, lables, test_size= 0.20, random_state=42)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', trainX.shape)
print('Training labels shape: ', trainY.shape)
print('Test data shape: ', testX.shape)
print('Test labels shape: ', testY.shape)

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['cat','dog']
num_classes = len(classes)
samples_per_class = 2

for y, cls in enumerate(classes):
    print(y)
    print(cls)
    idxs = np.flatnonzero(trainY == y)
    print(idxs)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    print(idxs)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(trainX[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

# Preprocessing: reshape the image data into rows
X_train = np.reshape(trainX, (trainX.shape[0], -1))
X_test = np.reshape(testX, (testX.shape[0], -1))

# As a sanity check, print out the shapes of the data
print('Training data shape: ', X_train.shape)
print('Test data shape: ', X_test.shape)

# Preprocessing: subtract the mean image
# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)
print(mean_image.shape)
print(mean_image[:10]) # print a few of the elements
plt.figure(figsize=(4,4))
plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # visualize the mean image
plt.show()
mean_image = mean_image.astype('uint8')
print('mean type is:',mean_image.dtype)

# second: subtract the mean image from train and test data
X_train -= mean_image
X_test -= mean_image

# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

## source: Stanforf University
def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  c = np.array([0, 1])
  pred_class = []
  for i in range(num_train):
    scores = X[i].dot(W)
    pred_class.append(c[np.argmax(scores)])
    #print('scores size:',scores.shape)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  print(pred_class)

  return loss, dW, pred_class

import time

# generate a random SVM weight matrix of small numbers
W = np.random.randn(3073, 2) * 0.0001

loss, grad,predicted_class = svm_loss_naive(W, X_train, trainY, 0.000005)
print('loss: %f' % (loss, ))

print(classification_report(trainY, predicted_class, target_names=le.classes_))