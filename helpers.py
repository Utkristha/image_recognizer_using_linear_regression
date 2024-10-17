import numpy as np
import copy

def sigmoid(z):
    sig = 1/(1+np.exp(-z))
    return sig

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0.0
    return w,b

def propagate(W,b,X,Y):
    #forward propagation
    m = X.shape[0]
    A = sigmoid(np.dot(W.T,X)+b)
    cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    #backward propagation
    dw = (np.dot(X,(A-Y).T))/m
    db = (np.sum(A-Y))/m

    cost = np.squeeze(np.array(cost))

    grads = {
        "dw" : dw,
        "db" : db}
    
    return grads,cost

def optimize(w,b,X,Y,num_of_iterations = 100,learning_rate = 0.009,print_cost = False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_of_iterations):
        grads,cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i % 100 == 0:
            costs.append(cost)

            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))

    params = {
        "w" : w,
        "b" : b
    }

    grads = {
        "dw" : dw,
        "db" :db
    }

    return params,grads,cost

def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m)) 
    w = w.reshape(X.shape[0],1)   #check if it works without this

    A = sigmoid(np.dot(w.T,X)+b)

    for i in range(A.shape[1]):
        if A[0,i] < 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

