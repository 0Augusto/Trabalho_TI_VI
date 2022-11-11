from statistics import correlation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import sys
import math
import time
#cross correlation indicate the most likely position of the template in the image 

def cross_correlation(image, template): 
    #image and template are 2D arrays 
    #image is the image to be searched 
    #template is the image to be found in the image 
    #returns the x and y coordinates of the top left corner of the template in the image 
    #returns -1, -1 if the template is not found 
    #returns -2, -2 if the template is larger than the image 
    #returns -3, -3 if the template is empty 
    #returns -4, -4 if the image is empty 
    #returns -5, -5 if the image or template is not 2D 

    #check if the image and template are 2D 
    if len(image.shape) != 2 or len(template.shape) != 2: 
        return -5, -5 

    #check if the image or template is empty 
    if image.size == 0: 
        return -4, -4 
    if template.size == 0: 
        return -3, -3 

    #check if the template is larger than the image 
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]: 
        return -2, -2 

    #perform the cross correlation 
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) 

    #return the top left corner of the template in the image 
    return max_loc