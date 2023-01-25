import numpy as np


def get_ft_mag_phase(img):
    fft_img = np.fft.fft2(img)
    fft_img = np.fft.fftshift(fft_img)
    fft_img_phase = np.angle(fft_img)
    fft_img_mag = np.abs(fft_img)
    return fft_img_mag,fft_img_phase


def fft_unsharp_box_filter(img,kernel_size):

    # test_box = np.ones(img.shape)
    crow,ccol = img.shape[0]//2,img.shape[1]//2
    # test_box[ccol:ccol+3,crow:crow+3] = kernel_size//2
    box_kernel = np.ones((kernel_size,kernel_size)) * 1/(kernel_size*kernel_size)
    # pad the kernel
    box_kernel = np.pad(box_kernel,((img.shape[0]//2,img.shape[0]//2-kernel_size+1),(img.shape[1]//2,img.shape[1]//2-kernel_size+1)),mode='constant',constant_values=0)
    print(box_kernel.shape)
 
    fft_kernel = np.fft.fft2(box_kernel)
    fft_img = np.fft.fft2(img)
    print(img.shape)
    fft_filtered_img = fft_img * fft_kernel
    filtered_img =np.fft.ifftshift(np.real( np.fft.ifft2(fft_filtered_img)))
    
    print(filtered_img)
    return filtered_img


def fft_high_boost_filter(img,kernel_size,k):
    blurred_img = fft_unsharp_box_filter(img,kernel_size)
    mask = img - blurred_img
    return img + k*mask
