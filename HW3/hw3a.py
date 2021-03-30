import math
import numpy as np
from download_mnist import load
import operator
import time

def load_dataset():
    # Loading the MNIST dataset
    data = {}
    x_train, x_test, y_train, y_test = load()
    data['x_train'], data['y_train'], data['x_test'], data['y_test'] = x_train, x_test, y_train, y_test
    data['x_train'] = data['x_train'].reshape(60000, 784)
    data['x_test'] = data['x_test'].reshape(10000, 784)
    data['x_train'] = data['x_train'].astype(float)
    data['x_test'] = data['x_test'].astype(float)

    # Transpose the training and test datasets
    data['x_train'], data['x_test'] = np.transpose(data['x_train']), np.transpose(data['x_test'])

    return data

def init_nn(data):
    parameters = {'n_X': data['x_train'].shape[0], 'n_h1': 200, 'n_h2': 50, 'n_y': 10, 'learning_rate': 0.01, 'reg': 1e-3}
    # Defining structure of the neural network

    # Initializing the weights and bias for each layer of the neural network
    W1 = 0.01 * np.random.randn(parameters['n_h1'], parameters['n_X'])
    b1 = np.zeros((parameters['n_h1'], 1))
    parameters['W1'], parameters['b1'] = W1, b1

    W2 = 0.01 * np.random.randn(parameters['n_h2'], parameters['n_h1']) 
    b2 = np.zeros((parameters['n_h2'], 1))
    parameters['W2'], parameters['b2'] = W2, b2

    W3 = 0.01 * np.random.randn(parameters['n_y'], parameters['n_h2']) 
    b3 = np.zeros((parameters['n_y'], 1))
    parameters['W3'], parameters['b3'] = W3, b3

    return parameters

def forward_propagation(parameters, data, m):
    f_prop = {}
    # Forward Propagation
    f_prop['Z1'] = np.dot(parameters['W1'], data['x_train']) + parameters['b1']
    f_prop['A1'] = np.maximum(0, f_prop['Z1'])      # ReLU

    f_prop['Z2'] = np.dot(parameters['W2'], f_prop['A1']) + parameters['b2']
    f_prop['A2'] = np.maximum(0, f_prop['Z2'])  # ReLU

    f_prop['Z3'] = np.dot(parameters['W3'], f_prop['A2']) + parameters['b3']
    f_prop['A3'] = np.exp(f_prop['Z3']) / np.sum(np.exp(f_prop['Z3']), axis=0, keepdims=True)     # Softmax

    # Computing the loss
    logprobs = -np.log(f_prop['A3'][data['y_train'], range(m)])
    data_loss = np.sum(logprobs)/m
    reg_loss = 0.5 * parameters['reg'] * (np.sum(np.square(parameters['W1'])) + np.sum(np.square(parameters['W2'])) + np.sum(np.square(parameters['W3'])))
    J = data_loss + reg_loss

    return f_prop, J

def back_propagation(parameters, data, f_prop, m):
    grads = {}
    # Back Propagation (Gradient Descent)
    grads['dZ3'] = f_prop['A3'] 
    grads['dZ3'][data['y_train'], range(m)] -= 1
    grads['dW3'] = np.dot(grads['dZ3'], np.transpose(f_prop['A2'])) / m
    grads['db3'] = np.sum(grads['dZ3'], axis=1, keepdims=True) / m

    grads['dZ2'] = np.dot(np.transpose(parameters['W3']), grads['dZ3'])
    grads['dZ2'][f_prop['Z2'] <= 0] = 0
    grads['dW2'] = np.dot(grads['dZ2'], np.transpose(f_prop['A1'])) / m
    grads['db2'] = np.sum(grads['dZ2'], axis=1, keepdims=True) / m

    grads['dZ1'] = np.dot(np.transpose(parameters['W2']), grads['dZ2'])
    grads['dZ1'][f_prop['Z1'] <= 0] = 0
    grads['dW1'] = np.dot(grads['dZ1'], np.transpose(data['x_train'])) / m
    grads['db1'] = np.sum(grads['dZ1'], axis=1, keepdims=True) / m

    grads['dW3'] += parameters['reg'] * parameters['W3']
    grads['dW2'] += parameters['reg'] * parameters['W2']
    grads['dW1'] += parameters['reg'] * parameters['W1']

    # Update parameters
    parameters['W1'] -= parameters['learning_rate'] * grads['dW1']
    parameters['b1'] -= parameters['learning_rate'] * grads['db1']
    parameters['W2'] -= parameters['learning_rate'] * grads['dW2']
    parameters['b2'] -= parameters['learning_rate'] * grads['db2']
    parameters['W3'] -= parameters['learning_rate'] * grads['dW3']
    parameters['b3'] -= parameters['learning_rate'] * grads['db3']

    return grads, parameters    

def train_model(data, parameters, iter, m):
    t = time.time()
    for i in range(1, 501):    
        f_prop, J = forward_propagation(parameters, data, m)
        print("Loss in iteration ", i, " is: ", J)
        grads, updated_parameters = back_propagation(parameters, data, f_prop, m)
    
    print("Training time: ", time.time() - t, " seconds.")

    return grads, updated_parameters

def test_model(data, parameters):
    # Test phase
    Z1 = np.dot(parameters['W1'], data['x_test']) + parameters['b1']
    A1 = np.maximum(0, Z1)      # ReLU
    # print(A1.shape)

    Z2 = np.dot(parameters['W2'], A1) + parameters['b2']
    A2 = np.maximum(0, Z2)  # ReLU
    # print(A2.shape)

    Z3 = np.dot(parameters['W3'], A2) + parameters['b3']
    A3 = np.exp(Z3) / np.sum(np.exp(Z3), axis=0, keepdims=True)     # Softmax
    # print(A3.shape)

    pred = np.argmax(A3, axis=0)

    print("Test accuracy: ", np.mean(pred == data['y_test']))

if __name__ == '__main__':
    data = load_dataset()
    parameters = init_nn(data)

    grads, updated_parameters = train_model(data, parameters, iter=500, m=data['x_train'].shape[1])
    test_model(data, updated_parameters)