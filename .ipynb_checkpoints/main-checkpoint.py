import numpy as np
from PIL import Image
from matplotlib import pyplot

im = Image.open("/home/utkristha/projects/python/mlearning/cat_recognizer/load.bmp")

pyplot.imshow(im)
pyplot.show()