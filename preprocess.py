import numpy as np
import json
import os
import argparse
import random
import gc
from PIL import Image

def fetch_filename(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in [".jpeg", ".jpg", ".png", ".bmp"]:
                L.append(os.path.join(root, file))
    return L

def noise(img, percent):

    mask = np.ones(np.shape(img))
    ncols = np.shape(img)[1]
    nnoise = int(ncols * percent)
    if img.ndim == 2:
        for i in xrange(np.shape(img)[0]):
            inds = np.random.permutation(ncols)[:nnoise]
            mask[i, inds] = np.zeros(nnoise)
    else:
        for i in xrange(np.shape(img)[0]):
            for j in xrange(np.shape(img)[2]):
                inds = np.random.permutation(ncols)[:nnoise]
                mask[i, inds, j] = np.zeros(nnoise)
    return np.multiply(img, mask)

def preprocessing_gray(params):

    fnames = fetch_filename(params["img_root"])
    patch_size = params["patch_size"]

    if len(fnames) <= params["train_size"]:
        img_dirs = fnames
    else:
        imgs_dir = random.sample(fnames, params["train_size"])

    train_x = []
    train_y = []
    count = 0
    for img_dir in imgs_dir:
        im = np.array(Image.open(img_dir).convert('L'), dtype=np.uint8)
        noise_im = noise(im, params["noise_percent"]).astype(np.uint8)
        count = count + 1
        for i in xrange(16):
            xc = random.randint(0, np.shape(im)[0] - patch_size - 1)
            yc = random.randint(0, np.shape(im)[1] - patch_size - 1)
            # print (xc, yc)
            train_x.append(im[xc: xc + patch_size, yc: yc + patch_size])
            train_y.append(noise_im[xc: xc + patch_size, yc: yc + patch_size])
        print count
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    np.save("train_x_gray", train_x)
    np.save("train_y_gray", train_y)

def preprocessing_color(params):

    fnames = fetch_filename(params["img_root"])
    patch_size = params["patch_size"]
    if len(fnames) <= params["train_size"]:
        imgs_dir = fnames
    else:
        imgs_dir = random.sample(fnames, params["train_size"])

    ndim = 3
    train_x = []
    train_y = []
    count = 0
    for img_dir in imgs_dir:
        im = np.array(Image.open(img_dir), dtype=np.uint8)
        count = count + 1
        if im.ndim != ndim:
            continue

        noise_im = noise(im, params["noise_percent"]).astype(np.uint8)

        for i in xrange(16):
            xc = random.randint(0, np.shape(im)[0] - patch_size - 1)
            yc = random.randint(0, np.shape(im)[1] - patch_size - 1)
            # print (xc, yc)
            train_x.append(im[xc: xc + patch_size, yc: yc + patch_size, :])
            train_y.append(noise_im[xc: xc + patch_size, yc: yc + patch_size, :])
        print count

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)

    np.save("train_x_color", train_x)
    np.save("train_y_color", train_y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_root', default="./dataset/", help='image folder')
    parser.add_argument('--train_size', default=5000, type=int, help='training data size')
    parser.add_argument('--patch_size', default=32, type=int, help="size of training patch")
    parser.add_argument('--noise_percent', default=0.8, type=float, help="missing pixel percent for image")

    args = parser.parse_args()
    params = vars(args)
    print json.dumps(params, indent=2)

    preprocessing_gray(params)
