# import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as im
from matplotlib.widgets import RectangleSelector

import numpy as np
#from PIL import Image
import math
import copy
import scipy
from math import sqrt
import cv2
from matplotlib.colors import Normalize
from skimage.data import shepp_logan_phantom 
from skimage.transform import radon , rescale , iradon
from phantominator import shepp_logan
import time

def bilinear_rotation_abdallah(angle):
# Converting degrees to radians
    angle = math.radians(angle)

    cosine = math.cos(angle)
    sine = math.sin(angle)

    # Define the height of the image
    oldWidth = self.imageTshape.shape[0]
    # Define the width of the image
    oldHeight = self.imageTshape.shape[1]

    # Initilize rotated image 
    rotatedImage = np.zeros((oldWidth,oldHeight)) 

    # Find the center of the Rotated T image
    centerHeight = int( (oldHeight+1)/2) # mid row
    centerWidth = int( (oldWidth+1)/2 ) # mid col

    for i in range(oldWidth):
        for j in range(oldHeight):
            x = -(j-centerWidth)*sine + (i-centerHeight)*cosine
            y = (j-centerWidth)*cosine + (i-centerHeight)*sine
            
            # Add offset
            x += centerHeight
            y += centerWidth

            # Calculate the coordinate values for 4 surrounding pixels.
            x_floor = math.floor(x)
            x_ceil = min(oldWidth-1, math.ceil(x))
            y_floor = math.floor( y )
            y_ceil = min(oldHeight - 1, math.ceil(y))
            
            if (x >= 0 and y >= 0 and x < oldWidth and y < oldHeight):
                if (x_ceil == x_floor) and (y_ceil == y_floor):
                    q = self.imageTshape[int(y), int(x)]
                elif (y_ceil == y_floor):
                    q1 = self.imageTshape[int(y), int(x_floor)]
                    q2 = self.imageTshape[int(y), int(x_ceil)]
                    q = q1 * (x_ceil - x) + q2 * (x - x_floor)
                elif (x_ceil == x_floor):
                    q1 = self.imageTshape[int(y_floor), int(x)]
                    q2 = self.imageTshape[int(y_ceil), int(x)]
                    q = (q1 * (y_ceil - y)) + (q2 * (y - y_floor))
                else:
                    p1 = self.imageTshape[y_floor, x_floor]
                    p2 = self.imageTshape[y_ceil, x_floor]
                    p3 = self.imageTshape[y_floor, x_ceil]
                    p4 = self.imageTshape[y_ceil, x_ceil]

                    q1 = p1 * (y_ceil - y) + p2 * (y - y_floor)
                    q2 = p3 * (y_ceil - y) + p4 * (y - y_floor)
                    q = q1 * (x_ceil - x) + q2 * (x - x_floor)

                rotatedImage[j][i] = q



def rotate_image_bilinear_matrix(img,angle):
    print(img)

    h,w = img.shape
    center= (h//2,w//2)

    new_img = np.zeros((h,w))
    i_pad= h/2
    j_pad =w/2

    rot_mat = np.array([[np.cos(angle),-np.sin(angle),0],
                        [np.sin(angle), np.cos(angle),0],
                        [0,0,1]])

    a = np.array([[1,0,-i_pad],
                  [0,1,-j_pad],
                  [0,0,1]])

    b = np.array([[1,0,i_pad],
                  [0,1,j_pad],
                  [0,0,1]])


    for i in range(0,h):
        for j in range(0,w):
            
            ij = np.array([i,j,1])

            x,y,_ = np.dot(np.dot(np.dot(b,rot_mat),a),ij)



            x_floor = math.floor(x)
            x_ceil = min(w-1, math.ceil(x))
            y_floor = math.floor( y )
            y_ceil = min(h - 1, math.ceil(y))
        
            if (x >= 0 and y >= 0 and x < h and y < w):
                if (x_ceil == x_floor) and (y_ceil == y_floor):
                    q = img[int(y), int(x)]
                elif (y_ceil == y_floor):
                    q1 = img[int(y), int(x_floor)]
                    q2 = img[int(y), int(x_ceil)]
                    q = q1 * (x_ceil - x) + q2 * (x - x_floor)
                elif (x_ceil == x_floor):
                    q1 = img[int(y_floor), int(x)]
                    q2 = img[int(y_ceil), int(x)]
                    q = (q1 * (y_ceil - y)) + (q2 * (y - y_floor))
                else:
                    p1 = img[y_floor, x_floor]
                    p2 = img[y_ceil, x_floor]
                    p3 = img[y_floor, x_ceil]
                    p4 = img[y_ceil, x_ceil]

                    q1 = p1 * (y_ceil - y) + p2 * (y - y_floor)
                    q2 = p3 * (y_ceil - y) + p4 * (y - y_floor)
                    q = q1 * (x_ceil - x) + q2 * (x - x_floor)

                new_img[i][j] = q

    # print(new_img)
    return new_img


def shear_0(img,shear_value):

    print(img)

    h,w = img.shape
    a = h/2
    b= w/2
    center= (h//2,w//2)

    new_img = np.zeros((h,w))
    for i in range(0,h):
        for j in range(0,w):
                x = (i - a)  + a
                y = -(i - a)  + (j - b) * shear_value + b
                if x < h and y < w and x >0 and y >0:
                    new_img[i][j] = img[int(x),int(y)]


    # print(new_img)
    return new_img


def rotate_image_bilinear(img,angle):

    print(img)

    h,w = img.shape
    a = h/2
    b= w/2
    center= (h//2,w//2)

    new_img = np.zeros((h,w))
    for i in range(0,h):
        for j in range(0,w):
                x = (i - a) * np.cos(angle) + (j - b) * np.sin(angle) + a
                y = -(i - a) * np.sin(angle) + (j - b) * np.cos(angle) + b
                if x < h and y < w and x >0 and y >0:
                    x_floor = math.floor(x)
                    x_ceil = min(w-1, math.ceil(x))
                    y_floor = math.floor( y )
                    y_ceil = min(h - 1, math.ceil(y))
                
                    if (x >= 0 and y >= 0 and x < h and y < w):
                        if (x_ceil == x_floor) and (y_ceil == y_floor):
                            q = img[int(x),int(y)]
                        elif (y_ceil == y_floor):
                            q1 = img[int(x_floor),int(y)]
                            q2 = img[int(x_ceil),int(y)]
                            q = q1 * (x_ceil - x) + q2 * (x - x_floor)
                        elif (x_ceil == x_floor):
                            q1 = img[int(x),int(y_floor)]
                            q2 = img[int(x),int(y_ceil)]
                            q = (q1 * (y_ceil - y)) + (q2 * (y - y_floor))
                        else:
                            p1 = img[x_floor,y_floor]
                            p2 = img[x_floor,y_ceil]
                            p3 = img[x_ceil,y_floor]
                            p4 = img[x_ceil,y_ceil]

                            q1 = p1 * (y_ceil - y) + p2 * (y - y_floor)
                            q2 = p3 * (y_ceil - y) + p4 * (y - y_floor)
                            q = q1 * (x_ceil - x) + q2 * (x - x_floor)

                        new_img[i][j] = q


    # print(new_img)
    return new_img


def rotate_image(img,angle):
    print(img)

    h,w = img.shape

    center= (h//2,w//2)

    new_img = np.zeros((h,w))
    i_pad= h/2
    j_pad =w/2
    for i in range(0,h):
        for j in range(0,w):

                a = np.array([[1,0,-i_pad],
                              [0,1,-j_pad],
                              [0,0,1]])

                b = np.array([[1,0,i_pad],
                              [0,1,j_pad],
                              [0,0,1]])

                rot_mat = np.array([[np.cos(angle),-np.sin(angle),0],
                                    [np.sin(angle), np.cos(angle),0],
                                    [0,0,1]])
                ij = np.array([i,j,1])

                x,y,_ = np.dot(np.dot(np.dot(b,rot_mat),a),ij)

                if x < h and y < w and x >0 and y >0:
                        new_img[i,j] = img[int(x),int(y)]

    # print(new_img)
    return new_img

def scale_image(img,scale_factor):
    print(img)

    h,w = img.shape

    center= (h//2,w//2)

    new_img = np.zeros((h,w))
    i_pad= h/2
    j_pad =w/2
    for i in range(0,h):
        for j in range(0,w):

                a = np.array([[1,0,-i_pad],
                              [0,1,-j_pad],
                              [0,0,1]])

                b = np.array([[1,0,i_pad],
                              [0,1,j_pad],
                              [0,0,1]])

                scale_mat = np.array([[1/scale_factor,0,0],
                                      [0,1/scale_factor,0],
                                      [0,0,1]])

                ij = np.array([i,j,1])

                x,y,_ = np.dot(np.dot(np.dot(b,scale_mat),a),ij)
    
                if x < h and y < w and x >0 and y >0:
                        new_img[i,j] = img[int(x),int(y)]
    return new_img

def shear_image(img,shear_factor):
    print(img)

    h,w = img.shape

    center= (h//2,w//2)

    new_img = np.zeros((h,w))
    i_pad= h/2
    j_pad =w/2
    for i in range(0,h):
        for j in range(0,w):

                a = np.array([[1,0,-i_pad],
                              [0,1,-j_pad],
                              [0,0,1]])

                b = np.array([[1,0,i_pad],
                              [0,1,j_pad],
                              [0,0,1]])
                shear_mat = np.array([[1,0,0],
                                      [shear_factor,1,0],
                                      [0,0,1]])
                ij = np.array([i,j,1])

                x,y,_ = np.dot(np.dot(np.dot(b,shear_mat),a),ij)
  
                if x < h and y < w and x >0 and y >0:
                        new_img[i,j] = img[int(x),int(y)]

    print(new_img)
    return new_img


def translate_image(img,rotation_angle=0,scale_factor=1,shear_factor =0):
    # if more than 1 translation applied the function will rotate then scale then shear
    print(img)

    h,w = img.shape
    new_img = np.zeros((h,w))
    i_pad= h/2
    j_pad =w/2

    # a,b are used to move the image center to the origin before rotating or scaling..,and then return the image to its center
    a = np.array([[1,0,-i_pad],
                  [0,1,-j_pad],
                  [0,0,1]])

    b = np.array([[1,0,i_pad],
                  [0,1,j_pad],
                  [0,0,1]])
    identity = np.array([[1,0,0],
                         [0,1,0],
                         [0,0,1]])

    scale_mat = np.array([[scale_factor,0,0],
                          [0,scale_factor,0],
                          [0,0,1]]) if scale_factor != 1 else identity
                          
    shear_mat = np.array([[1,0,0],
                          [shear_factor,1,0],
                          [0,0,1]]) if shear_factor != 0 else identity

    rot_mat = np.array([[np.cos(rotation_angle),-np.sin(rotation_angle),0],
                        [np.sin(rotation_angle), np.cos(rotation_angle),0],
                        [0,0,1]])if rotation_angle%(np.pi*2) != 0 else identity

    for i in range(0,h):
        for j in range(0,w):

                ij = np.array([i,j,1])
                
                # [b] . [rot_mat] . [scale_mat] . [shear_mat] . [a]
                x,y,_ = np.dot(np.dot(np.dot(np.dot(np.dot(b,rot_mat),scale_mat),shear_mat),a),ij)
                
                if x < h and y < w and x >0 and y >0:
                        new_img[i,j] = img[int(x),int(y)]

    return new_img

def rgb2gray(rgb):
    return np.dot(rgb, [0.299, 0.587, 0.144])

def bl_interpolation(img,scaling_factor):
    	#get dimensions of original image
        orig_img = img
        orig_length, orig_widthw = orig_img.shape[:2];
        #create an array of the desired shape. 
        #We will fill-in the values later.
        new_h = int(orig_length * scaling_factor)
        new_w = int(orig_widthw * scaling_factor)
        resized = np.zeros((int(new_h), int(new_w)))
        #Calculate horizontal and vertical scaling factor
        w_scale_factor = (orig_widthw ) / (new_w ) if new_h != 0 else 0
        h_scale_factor = (orig_length ) / (new_h ) if new_w != 0 else 0
        
        
        
        for i in range(new_h):
            for j in range(new_w):
                #map the coordinates back to the original image
                x = i * h_scale_factor
                y = j * w_scale_factor
                #calculate the coordinate values for 4 surrounding pixels.
                x_floor = math.floor(x)
                x_ceil = min( orig_length - 1, math.ceil(x))
                y_floor = math.floor(y)
                y_ceil = min(orig_widthw - 1, math.ceil(y))

                if (x_ceil == x_floor) and (y_ceil == y_floor):
                    q = orig_img[int(x), int(y)]
                elif (x_ceil == x_floor):
                    q1 = orig_img[int(x), int(y_floor)]
                    q2 = orig_img[int(x), int(y_ceil)]
                    q = q1 * (y_ceil - y) + q2 * (y - y_floor)
                elif (y_ceil == y_floor):
                    q1 = orig_img[int(x_floor), int(y)]
                    q2 = orig_img[int(x_ceil), int(y)]
                    q = (q1 * (x_ceil - x)) + (q2	 * (x - x_floor))
                else:
                    v1 = orig_img[x_floor, y_floor]
                    v2 = orig_img[x_ceil, y_floor]
                    v3 = orig_img[x_floor, y_ceil]
                    v4 = orig_img[x_ceil, y_ceil]

                    q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                    q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                    q = q1 * (y_ceil - y) + q2 * (y - y_floor)

                resized[i,j] = q
        return resized.astype(np.uint8)

def nn_interpolation(img, factor):
    newimg = np.zeros((img.shape[0] *factor,img.shape[1] *factor))
    for newimg_row_index in range(len(newimg[:,0])):
        for newimg_col_index in range(len(newimg[0,:])):
            if str(2* newimg_row_index/factor) in tuple('123456789') : # handel the ceiling of x.5 case
                oldimg_row_index = np.ceil(newimg_row_index/factor)
            else:
                oldimg_row_index = round(newimg_row_index/factor)
            if str(2* newimg_col_index/factor) in tuple('123456789') : # handel the ceiling of x.5 case
                oldimg_col_index = np.ceil(newimg_col_index/factor)
            else:
                oldimg_col_index =round(newimg_col_index/factor)
            # print('row_index : '+ str(newimg_row_index))
            # print('col_index : '+ str(newimg_col_index))
            # print('new_row_index : '+ str(oldimg_row_index))
            # print('new_col_index : '+ str(oldimg_col_index))
            newimg[newimg_row_index,newimg_col_index] = img[max(oldimg_row_index-1,0),max(oldimg_col_index-1,0)]
    return newimg

def t_img():

    # t_img = np.zeros((128,128))
    # horizontal_begining
    # hbar : horizontal bar , vbar: vertical bar
    # hbar_width = 70
    # hbar_height = vbar_width = 20
    # vbar_height = 50

    # hbar_xstart = (128 -hbar_width)//2
    # vbar_xstart = (128 -vbar_width)//2

    # top_padding = 30

    # t_img[  top_padding  :  top_padding+hbar_height      ,         hbar_xstart  :  hbar_xstart+hbar_width  ] = 255

    # t_img[  top_padding+hbar_height  :  top_padding+hbar_height+vbar_height    ,       vbar_xstart  :  vbar_xstart+vbar_width] =255
    # # t_img = translate_image(img,scale_factor=0.5)
    # t_img = shear_0(t_img,np.pi * 1/4)
    # # print('rotated image ......................')
    # print(t_img)


    # t_img = bl_interpolation(t_img,4)    
    # print('after interpolation..................')
    # print( t_img)
    # print(t_img.shape)
    pass

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
    cdf =np.array([])
    for i in range(len(distribution)):
        new_pixel_value = 255 * sum(distribution[0:i])
        cdf = np.append(cdf,round(new_pixel_value)) 
    new_img = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            new_img[i,j] = cdf[min(int(img[i,j]),255)]
    
    new_distribution = np.zeros((1,256)).flatten()
    for row in new_img:
        for pixel in row:
            new_distribution[min(int(pixel),255)] +=1

    return new_img, distribution*width *height, new_distribution

def test_unsharp_filter(img,kernel_size = 3,k=1):
    # input img must be in gray scale

    # kernel = np.ones((kernel_size,kernel_size)) * 1/(kernel_size*kernel_size)
    kernel = gkernel(l=kernel_size)
    # flip the kernel in x and y 
    # kernel = np.resize([i[::-1] for i in kernel][::-1],(kernel_size,kernel_size)) 
    print(kernel)
    new_img = np.pad(np.zeros_like(img),  kernel_size//2 , 'reflect')
    width, length = img.shape
    for row  in range(kernel_size//2,width-3):
        for column in range(kernel_size//2,length-3):
            sum = 0
            # np.sum(kernel * img[row:])
            for i in range(row-kernel_size//2,row + kernel_size//2+1):

                for j in range(column-kernel_size//2,column + kernel_size//2+1):

                    # print('i : '+str(i) + ' j : ' + str(j))
                    if i<kernel_size//2 or j <kernel_size//2:
                        continue
                    # print('sum = '+str(kernel[i%kernel_size][j%kernel_size]) + ' * '+str(img[i-kernel_size//2][j-kernel_size//2]))
                    sum += kernel[i%kernel_size][j%kernel_size] * img[i-kernel_size//2][j-kernel_size//2]

            # print(sum)
            new_img[row+2][column+2] = sum
    blur = new_img
    padded_original_img = np.pad(img,  kernel_size//2 , 'reflect')
    mask = padded_original_img - blur
    filtered_img = padded_original_img + k* mask

    # filtered_img = padded_original_img + k *( padded_original_img - new_img)

    img_min = min([min(arr) for arr in filtered_img])
    img_max = max([max(arr) for arr in filtered_img])
    print('img min : '+str(img_min) + 'img max :' + str(img_max)) 

    # if img_min <0:
    #     filtered_img += img_min
    # if img_max > 255 : 
    #     filtered_img *= 255/img_max
    filtered_img = filtered_img
    img_min = min([min(arr) for arr in filtered_img])
    img_max = max([max(arr) for arr in filtered_img])
    print('img min : '+str(img_min) + 'img max :' + str(img_max)) 


    return  filtered_img
def scale(g):
    # if img_min <0:
    #     filtered_img += img_min
    # if img_max > 255 : 
    #     filtered_img *= 255/img_max
    # gm = g - g.min()
    # gs = 255 * (gm / gm.max())
    # return gs
    l,w = g.shape
    for i in range(l):
        for j in range(w):
            if g[i,j] < 0:
                g[i,j] = 0
            if g[i,j] > 255:
                g[i,j] = 255
    return g
def gkernel(l=3, sig=2):
    """\
    Gaussian Kernel Creator via given length and sigma
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)

def old_unsharp_filter(img,kernel_size = 3,k=1):

    kernel = np.ones((kernel_size,kernel_size)) * 1/(kernel_size*kernel_size)
    # flip the kernel in x and y 
    kernel = np.resize([i[::-1] for i in kernel][::-1],(kernel_size,kernel_size)) 
    print(kernel)
    new_img = np.pad(np.zeros_like(img),  kernel_size//2 , 'reflect')
    width, length = img.shape
    for row  in range(kernel_size//2,width-3):
        for column in range(kernel_size//2,length-3):
            sum = 0
            for i in range(row-kernel_size//2,row + kernel_size//2+1):
                for j in range(column-kernel_size//2,column + kernel_size//2+1):

                    # print('i : '+str(i) + ' j : ' + str(j))
                    if i<kernel_size//2 or j <kernel_size//2:
                        continue
                    # print('sum = '+str(kernel[i%kernel_size][j%kernel_size]) + ' * '+str(img[i-kernel_size//2][j-kernel_size//2]))
                    sum += kernel[i%kernel_size][j%kernel_size] * img[i-kernel_size//2][j-kernel_size//2]

            # print(sum)
            new_img[row+2][column+2] = sum

    padded_original_img = np.pad(img,  kernel_size//2 , 'reflect')
    return  padded_original_img + k *( padded_original_img - new_img)

def spatial_box_filter(img,kernel_size,k=1):
    # input img must be in gray scale

    kernel = np.ones((kernel_size,kernel_size)) * 1/(kernel_size*kernel_size)
    # flip the kernel in x and y yet its not required for a box filter
    # kernel = np.resize([i[::-1] for i in kernel][::-1],(kernel_size,kernel_size)) 
    print(kernel)
    new_img = np.pad(np.zeros_like(img),  kernel_size//2 , 'reflect')
    width, length = img.shape
    for row  in range(kernel_size//2,width-kernel_size//2):
        for column in range(kernel_size//2,length-kernel_size//2):
            sum = 0
            # np.sum(kernel * img[row:])
            for i in range(row-kernel_size//2,row + kernel_size//2+1):

                for j in range(column-kernel_size//2,column + kernel_size//2+1):

                    # print('i : '+str(i) + ' j : ' + str(j))
                    if i<kernel_size//2 or j <kernel_size//2:
                        continue
                    # print('sum = '+str(kernel[i%kernel_size][j%kernel_size]) + ' * '+str(img[i-kernel_size//2][j-kernel_size//2]))
                    sum += kernel[i%kernel_size][j%kernel_size] * img[i-kernel_size//2][j-kernel_size//2]

            # print(sum)
            new_img[row+2][column+2] = sum
    blur = new_img
    # padded_original_img = np.pad(img,  kernel_size//2 , 'reflect')
    # mask = padded_original_img - blur
    # filtered_img = padded_original_img + k* mask

    # # filtered_img = padded_original_img + k *( padded_original_img - new_img)

    # img_min = min([min(arr) for arr in filtered_img])
    # img_max = max([max(arr) for arr in filtered_img])
    # print('img min : '+str(img_min) + 'img max :' + str(img_max)) 

    # filtered_img = scale_pixels_values(filtered_img)
    # img_min = min([min(arr) for arr in filtered_img])
    # img_max = max([max(arr) for arr in filtered_img])
    # print('img min : '+str(img_min) + 'img max :' + str(img_max)) 


    return  blur


def box_filter(img,kernel_size = 3,k=1):

    kernel = np.ones((kernel_size,kernel_size)) * 1/(kernel_size*kernel_size)
    # flip the kernel in x and y 
    kernel = np.resize([i[::-1] for i in kernel][::-1],(kernel_size,kernel_size)) 
    print(kernel)
    # make an empty img padded with reflect technique
    new_img = np.pad(np.zeros_like(img),  kernel_size//2 , 'reflect')
    width, length = new_img.shape
    # loop over the new img
    for row  in range(kernel_size//2,width):
        for column in range(kernel_size//2,length):
            sum = 0
            for i in range(row-kernel_size//2  , row + kernel_size//2+1):
                for j in range(column-kernel_size//2,column + kernel_size//2+1):

                    # print('i : '+str(i) + ' j : ' + str(j))
                    if i<kernel_size//2 or j <kernel_size//2:
                        continue
                    # print('sum = '+str(kernel[i%kernel_size][j%kernel_size]) + ' * '+str(img[i-kernel_size//2][j-kernel_size//2]))
                    sum += kernel[i%kernel_size][j%kernel_size] * img[i-kernel_size//2][j-kernel_size//2]

            # print(sum)
            new_img[row+2][column+2] = sum

           
    return new_img

def unsharp_filter(img,kernel_size = 3,k=1):
    # pad the original img then add the diffrence of the original and the blured to the original
    padded_original_img = np.pad(img,  kernel_size//2 , 'reflect')
    return  padded_original_img + k *( padded_original_img - box_filter(img))

def add_noise(img,min_loops = 300,max_loops = 10000):
    noise_img = copy.deepcopy(img)
    w , l = noise_img.shape

    #salt
    num_of_pixels = np.random.randint(min_loops, max_loops)
    for i in range(num_of_pixels):
        row_index=np.random.randint(0, w - 1)
        
        col_index=np.random.randint(0, l - 1)
        noise_img[row_index][col_index] = 255


    #pepper
    num_of_pixels = np.random.randint(min_loops , max_loops)
    for i in range(num_of_pixels):
        row_index= np.random.randint(0, w - 1)
        col_index= np.random.randint(0, l - 1)
        noise_img[row_index][col_index] = 0
        
    return noise_img

def median_filter(img, kernel_size):
    r,c = img.shape
    pixels_list = []
    ind = kernel_size // 2
    filtered_img = np.zeros((len(img),len(img[0])))
    for i in range(len(img)):
        for j in range(len(img[0])): 

            for z in range(kernel_size):
                # check if all of the kernel is in the image
                if i + z - ind < 0 or i + z - ind > len(img) - 1:
                    for c in range(kernel_size):
                        pixels_list.append(0)
                else:
                    if j + z - ind < 0 or j + ind > len(img[0]) - 1:
                        pixels_list.append(0)
                    else:
                        for k in range(kernel_size):
                            pixels_list.append(img[i + z - ind][j + k - ind])

            pixels_list.sort()
            filtered_img[i][j] = pixels_list[len(pixels_list) // 2]
            pixels_list = []
    return filtered_img

def get_ft_mag_phase(img):
    fft_img = np.fft.fft2(img)
    fft_img = np.fft.fftshift(fft_img)
    fft_img_phase = np.angle(fft_img)
    fft_img_mag = np.abs(fft_img)
    return fft_img_mag,fft_img_phase


def fft_box_filter(img,kernel_size):

    # test_box = np.ones(img.shape)
    crow,ccol = img.shape[0]//2,img.shape[1]//2
    # test_box[ccol:ccol+3,crow:crow+3] = kernel_size//2
    box_kernel = np.ones((kernel_size,kernel_size)) * 1/(kernel_size*kernel_size)
    # pad the kernel
    box_kernel = np.pad(box_kernel,((img.shape[0]//2,img.shape[0]//2-kernel_size),(img.shape[1]//2,img.shape[1]//2-kernel_size)),mode='constant',constant_values=0)
    print(box_kernel.shape)
 
    fft_kernel = np.fft.fft2(box_kernel)
    fft_img = np.fft.fft2(img)
    print(img.shape)
    fft_filtered_img = fft_img * fft_kernel
    filtered_img =np.fft.ifftshift(np.real( np.fft.ifft2(fft_filtered_img)))
    
    print(filtered_img)
    return filtered_img


def fft_unsharp_filter(img,kernel_size,k):
    blurred_img = fft_box_filter(img,kernel_size)
    mask = img - blurred_img
    return img + k* mask


def fourier_2(im,kernel_size):
    try:
        flag = False
        if im.shape[0] % 2 !=0:
            temp = np.zeros((im.shape[0]+1,im.shape[1]))
            temp[0:im.shape[0], 0:im.shape[1]] = im
            gray = temp
            im = temp
            img_data = temp
            flag = True
        if im.shape[1] % 2 != 0:
            temp = np.zeros((im.shape[0],im.shape[1]+1))
            temp[0:im.shape[0], 0:im.shape[1]] = im
            gray = temp
            im = temp
            img_data = temp
            flag = True
        if flag:
            img_fourier = fourier(img=gray, ret=True)
        else:
            img_fourier = fourier(ret=True)

        org_kernel, spat_filt_img = filter(kernel_size=kernel_size, ret=True, combo=1)
        kernel = np.zeros((im.shape[0], im.shape[1]))
        kernel[((img_fourier.shape[0] - kernel_size + 1) // 2):-((img_fourier.shape[0]-kernel_size) // 2), ((img_fourier.shape[1] - kernel_size +1) // 2):-((img_fourier.shape[1]-kernel_size) // 2)] = org_kernel
        kernel = scipy.fftpack.ifftshift(kernel)
        kernel_fourier = fourier(img=kernel, ret=True)
        freq_filtered = kernel_fourier * img_fourier
        freq_filtered = np.real(scipy.fftpack.ifft2(scipy.fftpack.fftshift(freq_filtered)))
        ifft_img.canvas.draw()
        diff_img = freq_filtered - spat_filt_img

        diff_ifft_img.axis.imshow(diff_img, cmap='gray')
        diff_ifft_img.canvas.draw()

        # the difference was 0 - a black image - which means that we got the same result but it was much faster,
        # simpler to code using the fourier transform and filtering in the frequency domain than it was in the
        # spatial domain.
    except Exception as e:
        logging.error(f'An error occured in the fourier_2 function (line 530) and it is : {e}')
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(f'{e}')
        msg.setWindowTitle('ERROR')
        msg.exec_()






def generate_phantom():   
    y,x = np.ogrid[-128: 128, -128: 128]
    omask = x**2+y**2 <= 64**2
    omask = 100*omask.astype('uint8')
    square = np.zeros((256,256)).astype('uint8')
    square[int(32):int(-1 * 32), int(32):int(-1 * 32)] = 100
    square2 = np.ones((256,256)).astype('uint8')
    square2 = square2 * 50
    final = square + omask + square2
    return final

def gen_noise(noises, img):
    try:
        noise = 0
        h, w = img.shape
        if not len(noises):
            return np.zeros((h,w))
        for kind in noises:
            if kind == 'gaussian':
                noise += np.random.normal(loc=0.0, scale=5, size=(h, w))
            if kind == 'uniform':
                noise += np.random.uniform(-10, 10, (h, w))
            if kind == 'rayleigh':
                noise += np.random.rayleigh(size=img.shape)
            if kind == 'exponential':
                noise += np.random.exponential(10, size=img.shape)
            if kind == 'gamma':
                noise += np.random.gamma(5, size=img.shape)
            if kind == 'salt_n_pepper':
                salt_percent = 0.1
                pepper_percent = 0.1
                percentile = np.random.rand(img.shape[0], img.shape[1])
                pepper = np.where(percentile < pepper_percent)
                salt = np.where(percentile < salt_percent)
                noise = img
                noise[salt] = 255
                noise[pepper] = 0
        noise = scale(noise)
        return noise
    except Exception as e:
        print(e)


def add_gaussian_uniform_noise(img,noise_type):
    noise = 0
    h, w = img.shape
    if noise_type == 'gaussian':
        noise += np.random.normal(loc=0.0, scale=5, size=(h, w))
    if noise_type == 'uniform':
        noise += np.random.uniform(-10, 10, (h, w))
    noise = scale(noise)
    return img + noise



def select_roi(self, eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    points = [x1, x2, y1, y2]
    for i in points:
        if i < 0:
            i = 0
        if i > 255:
            i = 255
    roi = self.noisy_img[int(min(points[2:])):int(max(points[2:])), int(min(points[:2])):int(max(points[:2]))]
    self.noisy_roi = roi
    self.roi.axis.imshow(roi, cmap='gray')
    self.roi.canvas.draw()
    self.histogram_equalization(roi)


def get_dist(img):
    distribution = np.zeros((1,256)).flatten()
    height,width = img.shape
    print(img.shape)
    # count the number of pixels for each color intensity
    sum = 0
    for row in img:
        for pixel in row:
            distribution[min(int(pixel),255)] +=1
            sum += pixel
            # print(pixel)

    mean = sum/(height*width-1)
    sum =0
    for row in img:
        for pixel in row:
            sum += (pixel-mean)**2
    standard_deviation = sqrt(np.sum/(height*width-1))
    print(mean)
    print(standard_deviation)
    return distribution,mean,standard_deviation



# img = np.random.randint(1,10,(5,5))
# print(img)
# print(unsharp_filter(img))
# print(img)

img = cv2.imread('R:\projects\image processing\images for testing\\boy.jpg')
cv_img = cv2.imread('R:\projects\image processing\images for testing\\finger_print.png')

# fig= plt.figure()
try:
    img = rgb2gray(img)
except:
    pass

# eq_img, old_dist, new_dist = histogram_equalization(img)
# # print(img-eq_img)
# # print(old_dist)
# # print(new_dist)
# fig = fig.figimage(img,cmap='gray')


# kernel_size = 3
# print(img.shape)
# sharped_img = test_unsharp_filter(img,kernel_size)
# print(sharped_img.shape)

# img = np.pad(img,  kernel_size//2 , 'reflect')
# fft_img_mag,fft_img_phase = get_ft_mag_phase(img)
# sp_img = spatial_box_filter(img,9)
# fft_sp_img_mag,fft_sp_img_phase = get_ft_mag_phase(sp_img)
# fr_img = fft_box_filter(img,9)
# fft_fr_img_mag,fft_fr_img_phase = get_ft_mag_phase(fr_img)

# noise_removed_img = remove_periodic_noise(img)
# noise_removed_img_mag,noise_removed_img_phase = get_ft_mag_phase(fr_img)


# unsharped_img = test_unsharp_filter(img,7,1)
# fft_unsharpped_img = fft_unsharp_filter(img,7,1)

# diff_img = scale(unsharped_img[6:,6:] - fft_unsharpped_img)
# filterfourier_filter(img,3)

# sharp_mask = img-blur_img 

kernel = np.array(
    [[0,1,1,1,0],
     [1,1,1,1,1],
     [1,1,1,1,1],
     [1,1,1,1,1],
     [0,1,1,1,0]])



def erode(img,kernel):
    
    convert_to_binary(img)
    # print(img)

    kernel_size = kernel.shape[0]
    original_img_size = img.shape
    # make an empty img padded with reflect technique
    img = np.pad(img,  kernel_size//2 , 'minimum')

    new_img = np.pad(np.zeros_like(img),  kernel_size//2 , 'reflect')
    width, length = new_img.shape
    # loop over the new img
    for row  in range(kernel_size//2,original_img_size[0]):
        for col in range(kernel_size//2,original_img_size[1]):
            # print(f'row {row}, col {col}')
            sum = 0
            hit = False
            match = False
                                
            val = kernel * img[row:row+kernel_size,col:col+kernel_size]
            # print(val)
            # print(img[row:row+kernel_size,col:col+kernel_size])
            # if val.any():
            #     # hit == True
            #     new_img[row,col] = 1
            if np.array_equal(val*True,kernel*True):
                match = True
                new_img[row,col] = 1
    return new_img



def dilate(img,kernel):
    kernel_size = kernel.shape[0]
    original_img_size = img.shape
    # make an empty img padded with reflect technique
    img = np.pad(img,  kernel_size//2 , 'minimum')

    new_img = np.pad(np.zeros_like(img),  kernel_size//2 , 'reflect')
    width, length = new_img.shape
    # loop over the new img
    for row  in range(kernel_size//2,original_img_size[0]):
        for col in range(kernel_size//2,original_img_size[1]):
            # print(f'row {row}, col {col}')
            sum = 0
            hit = False
            match = False
                                
            val = kernel * img[row:row+kernel_size,col:col+kernel_size]
            # print(val)
            # print(img[row:row+kernel_size,col:col+kernel_size])
            if val.any():
                # hit == True
                new_img[row,col] = 1

    return new_img

def convert_to_binary(img):
    for row  in range(img.shape[0]):
        for col in range(img.shape[1]):
            if img[row,col] !=0:
                img[row,col] =1

def open_img(img,kernel):
    return erode(dilate(img,kernel),kernel)

def close_img(img,kernel):
    return dilate(erode(img,kernel),kernel)

def morphological_filter(img):
    kernel = np.array(
    [[1,1,1],
     [1,1,1],
     [1,1,1]])
    return close_img(open_img(img,kernel),kernel)



# phantom = generate_phantom()
# g_noise = gen_noise(['gaussian'],phantom)
# u_noise = gen_noise(['uniform'],phantom)

# phantom = shepp_logan(256)
# print(phantom.size)

# for i in range(img.shape[0]):
#     for p in range(img.shape[1]):
#         img[i,p] = [img[i,p],img[i,p],img[i,p]]
# roi = cv2.selectROI(',',img)
# cv2.destroyAllWindows()
# roi_cropped = phantom[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]



# _, old_dist, _ = histogram_equalization(roi_cropped)
# print(old_dist)
# old_dist,mean,sd = get_dist(roi_cropped)
# print(old_dist)
# print(old_dist.shape)
# print(roi_cropped.shape)
# print(f"original shape {img.shape} erode {erode(img,kernel).shape} dilate {dilate(img,kernel).shape}")
fig, axes  = plt.subplots(1,3)
axes[0].imshow(img,cmap='gray')
axes[1].imshow(open_img(close_img(img,kernel),kernel),cmap='gray')
kernel = kernel.astype(np.uint8)
axes[2].imshow(morphological_filter(img),cmap='gray')

# axes[1,0].bar(np.arange(0,256),old_dist)
# axes[1,1].imshow(g_noise,cmap='gray')
# axes[1,2].imshow(roi_cropped,cmap='gray',norm= Normalize(vmin=0, vmax=255, clip=True))


# axes[1,2].imshow(noise_removed_img,cmap='gray')

# axes[2,0].imshow(np.log(1+fft_fr_img_mag),cmap='gray')
# axes[2,1].imshow(fft_fr_img_phase,cmap='gray')
# axes[2,2].imshow(np.fft.ifftshift(np.real( np.fft.ifft2(noise_removed_img))),cmap='gray')

# axes[3,0].imshow(scale(unsharped_img),cmap='gray')
# axes[3,1].imshow(scale(fft_unsharpped_img),cmap='gray')
# axes[3,2].imshow(diff_img,cmap='gray')
# np.real(np.fft.ifft2(np.fft.fft2(img)))

# axes[1,0].bar(np.arange(0,256),old_dist)
# axes[1,1].bar(np.arange(0,256),new_dist)

# # plt.bar(np.arange(0,256),old_dist)
# # plt.bar(np.arange(0,256),new_dist)


# phantom = shepp_logan(256)


# time.sleep(0.5)
# plt.close()
# fig, axes = plt.subplots(2, 2, figsize=(8, 4.5))

# axes[0,0].set_title("Original")
# axes[0,0].imshow(phantom, cmap=plt.cm.Greys_r)

# theta = np.linspace(0., 180., max(phantom.shape), endpoint=False)
# theta_b = list(range(0,180))
# theta_a = [0, 20, 40, 60, 160]
# print(theta)
# print(theta_a)
# print(theta_b)


# sinogram = radon(phantom, theta=theta)
# dx, dy = 0.5 * 180.0 / max(phantom.shape), 0.5 / sinogram.shape[0]
# axes[0,1].set_title("Radon transform\n(Sinogram)")
# axes[0,1].set_xlabel("Projection angle (deg)")
# axes[0,1].set_ylabel("Projection position (pixels)")
# axes[0,1].imshow(sinogram, cmap=plt.cm.Greys_r,
#         extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
#         aspect='auto')

# reconstruction_fbp = iradon(sinogram, theta=theta, filter_name = 'ramp')
# axes[1,0].imshow(reconstruction_fbp,cmap='gray')
# fig.tight_layout()
# plt.show()


plt.show()

# import cv2

# image = cv2.imread("flower.jpg")
# gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
# unsharp_image = cv2.addWeighted(image, 2.0, gaussian_3, -1.0, 0)
# cv2.imshow(unsharp_image)




