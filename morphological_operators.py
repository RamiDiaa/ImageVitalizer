import numpy as np
from misc import convert_to_binary

def erode(img,kernel):
    
    convert_to_binary(img)
    # print(img)

    kernel_size = kernel.shape[0]
    print(kernel_size)
    original_img_size = img.shape
    # make an empty img padded with reflect technique
    img = np.pad(img,  kernel_size//2 , 'minimum')

    new_img = np.pad(np.zeros_like(img),  kernel_size//2 , 'reflect')
    width, length = new_img.shape
    # loop over the new img
    for row  in range(kernel_size//2,original_img_size[0]):
        for col in range(kernel_size//2,original_img_size[1]):
                                
            val = kernel * img[row:row+kernel_size,col:col+kernel_size]
            if np.array_equal(val*True,kernel*True):
                # match = True
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
          
            val = kernel * img[row:row+kernel_size,col:col+kernel_size]
            if val.any():
                # hit == True
                new_img[row,col] = 1
    print(img.shape)
    return new_img



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

