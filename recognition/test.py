import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.filters import sobel, gaussian 

def show_img(image, title="Image", cmap_type="gray"):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

rocket_img = data.rocket()
rocket_gray_img = color.rgb2gray(rocket_img)
#show_img(rocket_gray_img)

#--------------------------------------------
#detect img edge
roket_egde = sobel(rocket_gray_img)
#show_img(roket_egde)

#--------------------------------------------
#gaussian method
roket_gaussian = gaussian(rocket_img)
show_img(roket_gaussian)
