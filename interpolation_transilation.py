import numpy as np
import math



def nn_interpolation(img, factor):
    newimg = np.zeros((int(img.shape[0] *factor),int(img.shape[1] *factor)))
    for newimg_row_index in range(len(newimg[:,0])):
        for newimg_col_index in range(len(newimg[0,:])):
            if str(2* newimg_row_index/factor) in tuple('123456789') : # handel the ceiling of x.5 case
                oldimg_row_index = np.floor(newimg_row_index/factor)
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

def bl_interpolation(img,scaling_factor):

    orig_img = img
    orig_length, orig_widthw = orig_img.shape[:2]
    
    new_h = int(orig_length * scaling_factor)
    new_w = int(orig_widthw * scaling_factor)
    resized = np.zeros((int(new_h), int(new_w)))
    w_scale_factor = (orig_widthw ) / (new_w ) if new_h != 0 else 0
    h_scale_factor = (orig_length) / (new_h ) if new_w != 0 else 0
    
    
    
    for i in range(new_h):
        for j in range(new_w):
            
            x = i * h_scale_factor
            y = j * w_scale_factor
        
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
    return resized#.astype(np.uint8)

def translate_image(img,rotation_angle=0,rotation_technique = 'bilinear',scale_factor=1,shear_factor =0):
    # if more than 1 translation applied the function will rotate then scale then shear
    # it will interpolate by bilinear interpolation by default
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

    scale_mat = np.array([[1/scale_factor,0,0],
                          [0,1/scale_factor,0],
                          [0,0,1]]) if scale_factor != 1 else identity
                          
    shear_mat = np.array([[1,0,0],
                          [shear_factor,1,0],
                          [0,0,1]]) if shear_factor != 0 else identity

    rot_mat = np.array([[np.cos(rotation_angle),np.sin(rotation_angle),0],
                        [-np.sin(rotation_angle), np.cos(rotation_angle),0],
                        [0,0,1]])if rotation_angle%(np.pi*2) != 0 else identity

    for i in range(0,h):
        for j in range(0,w):

            ij = np.array([i,j,1])
            
            # [b] . [rot_mat] . [scale_mat] . [shear_mat] . [a]
            x,y,_ = np.dot(np.dot(np.dot(np.dot(np.dot(b,rot_mat),scale_mat),shear_mat),a),ij)
            
            if x < h and y < w and x >0 and y >0:
                if rotation_technique == 'nearest neighbour':
                    new_img[i,j] = img[int(x),int(y)]
                else:
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
                            v1 = img[x_floor,y_floor]
                            v2 = img[x_floor,y_ceil]
                            v3 = img[x_ceil,y_floor]
                            v4 = img[x_ceil,y_ceil]

                            q1 = v1 * (y_ceil - y) + v2 * (y - y_floor)
                            q2 = v3 * (y_ceil - y) + v4 * (y - y_floor)
                            q = q1 * (x_ceil - x) + q2 * (x - x_floor)

                        new_img[i][j] = q
    return new_img
