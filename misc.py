import numpy as np


def convert_to_binary(img):
    for row  in range(img.shape[0]):
        for col in range(img.shape[1]):
            if img[row,col] !=0:
                img[row,col] =1


def scale_pixels_values(mat):
    #   scaling
    # if img_min <0:
    #     filtered_img += img_min
    # if img_max > 255 : 
    #     filtered_img *= 255/img_max
    # gm = mat - g.min()
    # gs = 255 * (gm / gm.max())
    # return gs
    
    # clipping
    l,w = mat.shape
    for i in range(l):
        for j in range(w):
            if mat[i,j] < 0:
                mat[i,j] = 0
            if mat[i,j] > 255:
                mat[i,j] = 255
    return mat

def rgb2gray(rgb):
    return np.dot(rgb, [0.299, 0.587, 0.144])


