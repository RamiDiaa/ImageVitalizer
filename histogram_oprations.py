import numpy as np
from math import sqrt


def histogram_equalization(img):
    distribution = np.zeros((1,256)).flatten()
    height,width = img.shape
    # count the number of pixels for each color intensity
    for row in img:
        for pixel in row:
            distribution[min(int(pixel),255)] +=1
            
    # divide over the total number of pixels to get probability instead of counts
    distribution /= width *height 

    # Cumulative Distribution Function 
    # where indices are the gray scale values an the values are the new pixel value
    cdf =np.array([])
    for i in range(len(distribution)):
        new_pixel_value = 255 * sum(distribution[0:i])
        cdf = np.append(cdf,round(new_pixel_value)) 
    new_img = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            # where indices of cdf array are the gray scale values an the values are the new pixel value
            new_img[i,j] = cdf[min(int(img[i,j]),255)]
    
    new_distribution = np.zeros((1,256)).flatten()
    for row in new_img:
        for pixel in row:

            new_distribution[min(int(pixel),255)] +=1

    return new_img, distribution*width *height, new_distribution





def get_distribution_stat(img):
    distribution = np.zeros((1,256)).flatten()
    height,width = img.shape
    print(img.shape)
    # count the number of pixels for each color intensity
    # num =0
    sum = 0
    for row in img:
        for pixel in row:
            distribution[min(int(pixel),255)] +=1
            sum += pixel
            # num +=1
            # print(pixel)

    mean = sum/(height*width)
    sum =0
    for row in img:
        for pixel in row:
            sum += (pixel-mean)**2
    standard_deviation = sqrt(sum/(height*width))
    print(mean)
    print(standard_deviation)
    return distribution,mean,standard_deviation
