import numpy as np



def add_salt_pepper_noise(img,min_loops = 300,max_loops = 10000):
    noise_img = img #copy.deepcopy(img)
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
    print(img - noise_img)
    return noise_img



def add_gaussian_uniform_noise(img,noise_type):
    noise = 0
    h, w = img.shape
    if noise_type == 'gaussian':
        noise = np.random.normal(loc=0.0, scale=5, size=(h, w))
    if noise_type == 'uniform':
        noise = np.random.uniform(-10, 10, (h, w))
    # noise = scale_pixels_values(noise)
    return img + noise


