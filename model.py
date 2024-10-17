from loading_data import *
from helpers import *

def run_model(X_train,Y_train,X_test,Y_test,num_of_Iteration = 2000,learning_rate = 0.5,print_cost = True):
    w,b = initialize_with_zeros(X_train.shape[0])
    print(X_train.shape[0])
    params,grads,costs = optimize(w,b,X_train,Y_train,num_of_iterations=num_of_Iteration,
                                  learning_rate=learning_rate,print_cost=True)

    w = params["w"]
    b = params["b"]

    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)

    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_of_Iteration}
    
    return d