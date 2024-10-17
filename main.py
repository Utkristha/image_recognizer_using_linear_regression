import numpy as np
from loading_data import data_loader
from model import *
import matplotlib.pyplot as plt
from PIL import Image

train_set_x_original, train_set_y_original, test_set_x_original, test_set_y_original,classes  = data_loader()

#gives the total number of testing,training and number of pixels in training data
training_number = train_set_x_original.shape[0]
testing_number = test_set_x_original.shape[0]
num_px = train_set_x_original.shape[1]

#flattening our data set 
train_set_x_flatten = train_set_x_original.reshape(train_set_x_original.shape[0],-1).T
test_set_x_flatten = test_set_x_original.reshape(test_set_x_original.shape[0],-1).T

#normalizing our data set to be from range [0,1] as rgb ranges from [0,255]
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

logistic_regression_model = run_model(train_set_x,train_set_y_original,test_set_x,test_set_y_original)


my_image = "dog1.jpg"   

fname = "images/" + my_image
image = np.array(Image.open(fname).resize((num_px, num_px)))
plt.imshow(image)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(logistic_regression_model["w"], logistic_regression_model["b"], image)

print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
plt.show()