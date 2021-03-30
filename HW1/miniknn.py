import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



# load mini training data and labels
mini_train = np.load('knn_minitrain.npy')
mini_train_label = np.load('knn_minitrain_label.npy')

# randomly generate test data
mini_test = np.random.randint(20, size=20)
mini_test = mini_test.reshape(10,2)

# for i in range(len(mini_train)):
    # print(mini_train[i], mini_train_label[i])
# Define knn classifier
def kNNClassify(newInput, dataSet, labels, k):
    result=[]
    ########################
    # Input your code here #
    ########################

    # create a 2D array of zeros to store the L2 distances
    dist = np.zeros((40, 10))
    # run for all samples in the train dataset
    for i in range(len(dataSet)):
        # compute the L2 distance for each training sample and test sample
        d = np.sqrt(np.sum((newInput - dataSet[i, :])**2, axis=1))
        # store each resulting value in each row of the distance matrix
        dist[i, :] = d

    # in order to find the k-smallest distances over each row, transpose the matrix
    dist = np.transpose(dist)

    for i in range(len(dist)):
        # obtains the k-smallest distances' indices 
        indices = np.argsort(dist[i])[:k]
        # create an array to store the closest class to the test sample
        classes = [0] * 4
        # for each index in the indices 
        for j in range(len(indices)):
            # find the label corresponding to that index
            label = mini_train_label[indices[j]]
            # increment that class by 1, indicating the class for the test sample
            classes[label] += 1
        # add the most frequent class to result
        result.append(np.argmax(classes))        
    
    ####################
    # End of your code #
    ####################
    return result

outputlabels=kNNClassify(mini_test,mini_train,mini_train_label,9)

print ('random test points are:', mini_test)
print ('knn classfied labels for test:', outputlabels)

# plot train data and classfied test data
train_x = mini_train[:,0]
train_y = mini_train[:,1]
fig = plt.figure()
plt.scatter(train_x[np.where(mini_train_label==0)], train_y[np.where(mini_train_label==0)], color='red')
plt.scatter(train_x[np.where(mini_train_label==1)], train_y[np.where(mini_train_label==1)], color='blue')
plt.scatter(train_x[np.where(mini_train_label==2)], train_y[np.where(mini_train_label==2)], color='yellow')
plt.scatter(train_x[np.where(mini_train_label==3)], train_y[np.where(mini_train_label==3)], color='black')

test_x = mini_test[:,0]
test_y = mini_test[:,1]
outputlabels = np.array(outputlabels)
plt.scatter(test_x[np.where(outputlabels==0)], test_y[np.where(outputlabels==0)], marker='^', color='red')
plt.scatter(test_x[np.where(outputlabels==1)], test_y[np.where(outputlabels==1)], marker='^', color='blue')
plt.scatter(test_x[np.where(outputlabels==2)], test_y[np.where(outputlabels==2)], marker='^', color='yellow')
plt.scatter(test_x[np.where(outputlabels==3)], test_y[np.where(outputlabels==3)], marker='^', color='black')

#save diagram as png file
plt.savefig("miniknn.png")