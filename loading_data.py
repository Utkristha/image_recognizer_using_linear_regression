import h5py
import numpy as np

def data_loader():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5','r')
    test_dataset = h5py.File('datasets/test_catvnoncat.h5','r')

    train_set_x_original = np.array(train_dataset["train_set_x"][:])
    train_set_y_original = np.array(train_dataset["train_set_y"][:])
    
    test_set_x_original = np.array(test_dataset['test_set_x'][:])
    test_set_y_original = np.array(test_dataset['test_set_y'][:])

    train_set_y_original = train_set_y_original.reshape((1,train_set_y_original.shape[0]))
    test_set_y_original = test_set_y_original.reshape((1,test_set_y_original.shape[0]))

    classes = np.array(test_dataset["list_classes"][:]) 

    return train_set_x_original,train_set_y_original,test_set_x_original,test_set_y_original,classes