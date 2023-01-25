import numpy as np

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


def spatial_unsharp_box_filter(img,kernel_size,k=1):
    # input img must be in gray scale

    kernal = np.ones((kernel_size,kernel_size)) * 1/(kernel_size*kernel_size)
    # flip the kernal in x and y yet its not required for a box filter
    # kernal = np.resize([i[::-1] for i in kernal][::-1],(kernal_size,kernal_size)) 
    print(kernal)
    new_img = np.pad(np.zeros_like(img),  kernel_size//2 , 'reflect')
    width, length = img.shape
    for row  in range(kernel_size//2,width-kernel_size//2):
        for column in range(kernel_size//2,length-kernel_size//2):
            sum = 0
            # np.sum(kernal * img[row:])
            for i in range(row-kernel_size//2,row + kernel_size//2+1):

                for j in range(column-kernel_size//2,column + kernel_size//2+1):

                    # print('i : '+str(i) + ' j : ' + str(j))
                    if i<kernel_size//2 or j <kernel_size//2:
                        continue
                    # print('sum = '+str(kernal[i%kernal_size][j%kernal_size]) + ' * '+str(img[i-kernal_size//2][j-kernal_size//2]))
                    sum += kernal[i%kernel_size][j%kernel_size] * img[i-kernel_size//2][j-kernel_size//2]

            # print(sum)
            new_img[row+2][column+2] = sum
    return  new_img #blurred image



def spatial_high_boost_filter(img,kernel_size,k):
    blur = spatial_unsharp_box_filter(img,kernel_size)
    padded_original_img = np.pad(img,  kernel_size//2 , 'reflect')
    mask = padded_original_img - blur
    high_boosted_img = padded_original_img + k* mask
    return high_boosted_img
