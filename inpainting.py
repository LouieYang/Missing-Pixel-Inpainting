import argparse
import time
from PIL import Image, ImageChops
import tensorflow as tf
import json
import numpy as np
import os
import sys
from scipy.ndimage.filters import gaussian_filter
from inpainting_cnn import *

PATCH_SIZE = 32

def is_greyscale(im):
    """
    Check if image is monochrome (1 channel or 3 identical channels)
    """
    if im.mode not in ("L", "RGB"):
        raise ValueError("Unsuported image mode")

    if im.mode == "RGB":
        rgb = im.split()
        if ImageChops.difference(rgb[0],rgb[1]).getextrema()[1]!=0:
            return False
        if ImageChops.difference(rgb[0],rgb[2]).getextrema()[1]!=0:
            return False
    return True

def salting(img, noiseim):
    rang = np.where(noiseim != 0)
    img[rang] = noiseim[rang]
    return img

def three_channel_inpainting(sess, noiseim_placeholder, logits, noiseim, params):
    batch_size = params["batch_size"]
    nrows, ncols, nchannel = np.shape(noiseim)

    count = 0
    L = []
    start_index = []
    im_denoised = np.zeros((nrows, ncols, nchannel), dtype=np.float32)
    im_accessed = np.zeros((nrows, ncols, nchannel), dtype=np.float32)

    for i in xrange(nrows - PATCH_SIZE + 1):
        for j in xrange(ncols - PATCH_SIZE + 1):
            L.append(noiseim[i: i + PATCH_SIZE, j: j + PATCH_SIZE, :])
            start_index.append((i, j))
            count = count + 1

            if count == batch_size or (i == nrows - PATCH_SIZE and j == ncols - PATCH_SIZE):
                input_ims = np.asarray(L) / 255.
                netinput = np.zeros((batch_size, PATCH_SIZE, PATCH_SIZE, nchannel))
                netinput[:count, :, :, :] = input_ims
                patch_denoised = sess.run(logits, feed_dict={noiseim_placeholder: netinput})

                for k in xrange(count):
                    start_r, start_c = start_index[k]
                    im_denoised[start_r: start_r + PATCH_SIZE, start_c: start_c + PATCH_SIZE, :] = im_denoised[start_r: start_r + PATCH_SIZE, start_c: start_c + PATCH_SIZE, :] + patch_denoised[k, :, :, :]
                    im_accessed[start_r: start_r + PATCH_SIZE, start_c: start_c + PATCH_SIZE, :] = im_accessed[start_r: start_r + PATCH_SIZE, start_c: start_c + PATCH_SIZE, :] + np.ones((PATCH_SIZE, PATCH_SIZE, nchannel))
                L = []
                start_index = []
                count = 0;

    im_denoised = im_denoised / im_accessed
    res = np.array(im_denoised) * 255.
    return res

def inpaintingCNN(params):
    im = Image.open(params['img_dir'])
    if not is_greyscale(im):
        noiseim = np.array(im.convert('RGB')).astype(np.float32)
        noiseim_placeholder = tf.placeholder(tf.float32, shape=[params['batch_size'], PATCH_SIZE, PATCH_SIZE, 3])
        logits = inference_color(noiseim_placeholder)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()
        saver.restore(sess, './color-model/INPATINGCNN-' + str(params['model_version']) + '.ckpt')
        res = three_channel_inpainting(sess, noiseim_placeholder, logits, noiseim, params)
        return salting(res.astype(np.uint8), noiseim)
    else:
        noiseim = np.array(im.convert('L')).astype(np.float32)

        nrows, ncols = np.shape(noiseim)
        noiseim = noiseim.reshape((nrows, ncols, 1))


        noiseim_placeholder = tf.placeholder(tf.float32, shape=[params['batch_size'], PATCH_SIZE, PATCH_SIZE, 1])
        logits = inference_gray(noiseim_placeholder)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()
        saver.restore(sess, './gray-model/INPATINGCNN-' + str(params['model_version']) + '.ckpt')

        res = three_channel_inpainting(sess, noiseim_placeholder, logits, noiseim, params)
        res = salting(res.astype(np.uint8), noiseim)
        return res.reshape((nrows, ncols))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
    parser.add_argument('--model_version', default=100000, type=int, help='version of model used')
    parser.add_argument('--img_dir', required=True, help='noise image directory')

    args = parser.parse_args()
    params = vars(args)
    print json.dumps(params, indent=2)
    Image.fromarray(inpaintingCNN(params)).save(os.path.splitext(params['img_dir'])[0] + "_inpainted.png")
