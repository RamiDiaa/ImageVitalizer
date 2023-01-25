import numpy as np 
from skimage.transform import radon , rescale , iradon



def back_projection(img,theta,filter_name = None):
    # theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    # theta_b = list(range(0,180))
    # theta_a = [0, 20, 40, 60, 160]

    sinogram = radon(img, theta=theta)
    dx, dy = 0.5 * 180.0 / max(img.shape), 0.5 / sinogram.shape[0]

    
    reconstruction_fbp = iradon(sinogram, theta=theta, filter_name = filter_name)

    return sinogram, dx,dy, reconstruction_fbp



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

