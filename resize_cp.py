import os
import shutil
from PIL import Image


# rm -rf dirname; mkdir dirname
def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


# resize image to 299*299, fill blank area with (147, 147, 147)
def resize_cp(img_path, size, output_path):
    img = Image.open(img_path)
    width, height = img.size
    fit_len = width if width > height else height
    new_img = Image.new('RGB', [fit_len, fit_len], (147, 147, 147))
    x = int((fit_len - width) / 2)
    y = int((fit_len - height) / 2)
    new_img.paste(img, (x, y, x + width, y + height))
    new_img.resize(size=size).save(output_path)


# get image names of different type
train_imgnames = os.listdir('train')
test_imgnames = os.listdir('test')
train_cat = filter(lambda x: x[:3] == 'cat', train_imgnames)
train_dog = filter(lambda x: x[:3] == 'dog', train_imgnames)

rmrf_mkdir('train_classified')
rmrf_mkdir('test_processed')
os.mkdir('train_classified\\cat')
os.mkdir('train_classified\\dog')
os.mkdir('test_processed\\all')

img_size = [299, 299]

for filename in train_cat:
    resize_cp('train\\' + filename, img_size,
              'train_classified\\cat\\' + filename)
for filename in train_dog:
    resize_cp('train\\' + filename, img_size,
              'train_classified\\dog\\' + filename)
for filename in test_imgnames:
    resize_cp('test\\' + filename, img_size,
              'test_processed\\all\\' + filename)
