import math
import numpy as np  
from download_mnist import load
import operator  
import time
# classify using kNN  
#x_train = np.load('../x_train.npy')
#y_train = np.load('../y_train.npy')
#x_test = np.load('../x_test.npy')
#y_test = np.load('../y_test.npy')
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28,28)
x_test  = x_test.reshape(10000,28,28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)
def kNNClassify(newInput, dataSet, labels, k): 
    result=[]
    # print(max(y_train))
    ########################
    # Input your code here #
    ########################
    # create a 2D array of zeros to store the L2 distances
    dist = np.zeros((len(dataSet), len(newInput)))
    # run for all samples in the train dataset
    for i in range(len(dataSet)):
        for j in range(len(newInput)):
            # compute the L2 distance for each training sample and test sample
            d = np.sqrt(np.sum((newInput[j] - dataSet[i])**2))
            # store each resulting value in each row of the distance matrix
            dist[i, j] = d

    # in order to find the k-smallest distances over each row, transpose the matrix
    dist = np.transpose(dist)

    for i in range(len(dist)):
        # obtains the k-smallest distances' indices from each row of the dist array
        indices = np.argsort(dist[i])[:k]
        # create an array to store the closest class to the test sample
        classes = [0] * 10
        # for each index in the indices 
        for j in range(len(indices)):
            # find the label corresponding to that index
            label = labels[indices[j]]
            # increment that class by 1, indicating the class for the test sample
            classes[label] += 1
        # add the most frequent class to result
        result.append(np.argmax(classes))
    
    
    ####################
    # End of your code #
    ####################
    return result

start_time = time.time()
outputlabels=kNNClassify(x_test[0:25],x_train,y_train,9)
result = y_test[0:25] - outputlabels
result = (1 - np.count_nonzero(result)/len(outputlabels))
print ("---classification accuracy for knn on mnist: %s ---" %result)
print ("---execution time: %s seconds ---" % (time.time() - start_time))