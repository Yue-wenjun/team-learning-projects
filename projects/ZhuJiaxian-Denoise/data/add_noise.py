import numpy as np
import cv2
import os
os.chdir(os.path.dirname(__file__))

def gaussianNoise(image:np.ndarray, mean, sigma):
    '''给图片添加高斯噪声, 返回的图片类型为 uint8'''
    gauss = np.random.normal(mean,sigma,image.shape)
    noisy_img = image.astype(np.float32) + gauss
    return noisy_img.clip(0, 255).astype(np.uint8)

def poisonNoise(image:np.ndarray):
    '''给图片添加泊松噪声'''
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy_img = np.random.poisson(image * vals) / float(vals)
    return noisy_img.clip(0, 255).astype(np.uint8)

def pepper_salt_noise(image:np.ndarray, s_vs_p=0.5, amount = 0.04):
    '''给图片添加椒盐噪声'''
    noisy_img = np.copy(image)
    #添加salt噪声
    num_salt = np.ceil(amount * image.size * s_vs_p)
    #设置添加噪声的坐标位置
    coords = [np.random.randint(0,i - 1, int(num_salt)) for i in image.shape]
    noisy_img[coords[0],coords[1],:] = [255,255,255]
    #添加pepper噪声
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    #设置添加噪声的坐标位置
    coords = [np.random.randint(0,i - 1, int(num_pepper)) for i in image.shape]
    noisy_img[coords[0],coords[1],:] = [0,0,0]

    return noisy_img

if __name__ == '__main__':
    for name in os.listdir('original/'):
        img = cv2.imread('original/' + name)
        assert img is not None, f'Error reading image {name}.'
        noisy = gaussianNoise(img, 20, 20)
        cv2.imwrite('noisy/'+name, noisy)

