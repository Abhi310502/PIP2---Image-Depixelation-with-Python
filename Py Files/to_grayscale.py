import numpy as np
from a2_ex1 import to_grayscale
from PIL import Image
from glob import glob
from os import path
from torchvision import transforms


def grayscale(image_list, path_save):
    for i, path_ in enumerate(image_list):
        im = Image.open(path_)
        im_shape = 64
        transform = transforms.Compose([transforms.Resize(size=im_shape), transforms.CenterCrop(size=(im_shape, im_shape))])
        image = transform(im)
        image = np.array(image)
        image = (to_grayscale(image)).squeeze(0)
        image_new = Image.fromarray(image)
        path_ = path_save + "\\1" + str(i) + ".jpg"
        image_new.save(path_)


path_train = r"D:\Python\images"
path_save = r"D:\grayscale"
image_files = sorted(path.abspath(f) for f in glob(path.join(path_train, "**", "*.jpg"), recursive=True))
grayscale(image_files, path_save)
