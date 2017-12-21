import os
import numpy as np
import random
from PIL import Image
import cv2


# CLAHE function for PIL image
def pil_clahe(img, grid_size):
    lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size, grid_size))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


# add salt and pepper noise
def salt_pepper(img, SNR):
    arr = np.array(img)
    noise_n = int((1 - SNR) * arr.shape[0] * arr.shape[1])
    for i in range(noise_n):
        x = random.randint(0, arr.shape[0] - 1)
        y = random.randint(0, arr.shape[1] - 1)
        if random.randint(0, 1) == 0:
            arr[x, y] = [0, 0, 0]
        else:
            arr[x, y] = [255, 255, 255]
    return Image.fromarray(arr)


# hist_eq & flip_left_right & salt_pepper
def enhance_img(dir_in, dir_out, clahe=False, flip_lr=False, salt=None):
    img_names = os.listdir(dir_in)
    for img_name in img_names:
        basename = img_name[0:-4]
        img = Image.open('{0}\\{1}'.format(dir_in, img_name))
        if clahe:
            img = pil_clahe(img, 8)
        if salt:
            img = salt_pepper(img, salt)
        if flip_lr:
            img.transpose(Image.FLIP_LEFT_RIGHT) \
                .save('{0}\\{1}.flip.jpg'.format(dir_out, basename))
        img.save('{0}\\{1}'.format(dir_out, img_name))


enhance_img('train_classified\\cat', 'train_classified\\cat', flip_lr=True)
enhance_img('train_classified\\dog', 'train_classified\\dog', flip_lr=True)
