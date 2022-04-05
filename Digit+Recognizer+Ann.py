# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:28:44 2022

@author: Begy
"""

import numpy as np
import matplotlib.pyplot as plt

X = np.load("sign_langeage_data/X.npy")
Y = np.load("sign_langeage_data/Y.npy")





from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.18, random_state=35)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]


X_train_f = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])
X_test_f = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])


X_train = X_train_f.T
X_test = X_test_f.T
Y_train = Y_train.T
Y_test = Y_test.T


def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head


def initialize_parameters(X_train, Y_train):
    parameters = {"layer1_weights": np.random.randn(3,X_train.shape[0]) * 0.1,
                  "bias1": np.zeros((3,1)),
                  "layer2_weights": np.random.randn(Y_train.shape[0],3) * 0.1,
                  "bias2": np.zeros((Y_train.shape[0],1))}
    return parameters

def forward_propagation(X_train, parameters):

    layer1_predictions = np.tanh(np.dot(parameters["layer1_weights"],X_train) +parameters["bias1"])
    layer2_predictions = sigmoid(np.dot(parameters["layer2_weights"],layer1_predictions) + parameters["bias2"])

    cache = {
             "P1": layer1_predictions, 
             "P2": layer2_predictions
             }
    
    return layer2_predictions, cache


def compute_cost(P2, Y, parameters):
    logprobs = np.multiply(np.log(P2),Y)
    cost = -np.sum(logprobs)/Y.shape[1]
    return cost


def backward_propagation(parameters, cache, X, Y):
    dW2 = np.dot(cache["P2"]-Y,cache["P1"].T)/X.shape[1]
    db2 = np.sum(cache["P2"]-Y,axis =1,keepdims=True)/X.shape[1]
    dW1 = np.dot(np.dot(parameters["layer2_weights"].T,cache["P2"]-Y)*(1 - np.power(cache["P1"], 2)),X.T)/X.shape[1]
    db1 = np.sum(np.dot(parameters["layer2_weights"].T,cache["P2"]-Y)*(1 - np.power(cache["P1"], 2)),axis =1,keepdims=True)/X.shape[1]
    grads = {"dweight1": dW1,
             "dbias1": db1,
             "dweight2": dW2,
             "dbias2": db2}
    return grads

def update_parameters(parameters, grads, learning_rate = 0.01):
    parameters = {"layer1_weights": parameters["layer1_weights"]-learning_rate*grads["dweight1"],
                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],
                  "layer2_weights": parameters["layer2_weights"]-learning_rate*grads["dweight2"],
                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}
    
    return parameters


def predict(parameters,X_test):
    # x_test is a input for forward propagation
    P2, cache = forward_propagation(X_test,parameters)
    Y_prediction = np.zeros((1,X_test.shape[1]))   
    for i in range(P2.shape[1]):
        if P2[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction



def two_layer_neural_network(X_train, Y_train,X_test,Y_test, num_iterations):
    cost_list = []
    index_list = []
    
    parameters = initialize_parameters(X_train, Y_train)

    for i in range(0, num_iterations):
        P2, cache = forward_propagation(X_train,parameters)
        cost = compute_cost(P2, Y_train, parameters)
        grads = backward_propagation(parameters, cache, X_train, Y_train)
        parameters = update_parameters(parameters, grads)
        
        if i % 100 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print ("iteration %i Cost: %f" %(i, cost))
    plt.plot(index_list,cost_list)
    plt.xticks(index_list,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    
    # predict
    y_prediction_test = predict(parameters,X_test)
    y_prediction_train = predict(parameters,X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - Y_test)) * 100))
    return parameters

parameters = two_layer_neural_network(X_train, Y_train,X_test,Y_test, num_iterations=3000)

