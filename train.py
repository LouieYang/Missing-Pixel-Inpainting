import argparse
import numpy as np
import tensorflow as tf
import json
import os
import time
from inpainting_cnn import *

global im, noiseims
global idx
global train_gray

idx = 0

def load_mscoco_patches():
    global train_gray
    if train_gray:
        train_x = np.load("train_x_gray.npy")
        N, nrows, ncols = np.shape(train_x)
        train_x = train_x.reshape((N, nrows, ncols, 1))
        train_y = np.load("train_y_gray.npy")
        train_y = train_y.reshape((N, nrows, ncols, 1))
    else:
        train_x = np.load("train_x_color.npy")
        train_y = np.load("train_y_color.npy")

    return (train_x, train_y)

def fill_feed_dict(batch_size, im_placeholder, noiseim_placeholder):
    global idx, im, noiseims, train_gray

    if not train_gray:
        feed_dict = {
            noiseim_placeholder: noiseims[idx: idx + batch_size, :, :, :].astype(np.float32) / 255.,
            im_placeholder: im[idx: idx + batch_size, :, :, :].astype(np.float32) / 255.
        }
        idx = idx + batch_size

        if idx + batch_size >= np.shape(im)[0]:
            perm = np.random.permutation(np.shape(im)[0])
            im = im[perm, :, :, :]
            noiseims = noiseims[perm, :, :, :]
            idx = 0
    else:
        feed_dict = {
            noiseim_placeholder: noiseims[idx: idx + batch_size, :, :, :].astype(np.float32) / 255.,
            im_placeholder: im[idx: idx + batch_size, :, :, :].astype(np.float32) / 255.
        }
        idx = idx + batch_size

        if idx + batch_size >= np.shape(im)[0]:
            perm = np.random.permutation(np.shape(im)[0])
            im = im[perm, :, :, :]
            noiseims = noiseims[perm, :, :, :]
            idx = 0

    return feed_dict

def train(params, log_dir):

    global im, noiseims
    im, noiseims = load_mscoco_patches()
    with tf.Graph().as_default():

        if train_gray:
            im_placeholder = tf.placeholder(tf.float32, shape=[params['batch_size'], 32, 32, 1])
            noiseim_placeholder = tf.placeholder(tf.float32, shape=[params['batch_size'], 32, 32, 1])
            logits = inference_gray(noiseim_placeholder)
        else:
            im_placeholder = tf.placeholder(tf.float32, shape=[params['batch_size'], 32, 32, 3])
            noiseim_placeholder = tf.placeholder(tf.float32, shape=[params['batch_size'], 32, 32, 3])
            logits = inference_color(noiseim_placeholder)

        _loss = loss(logits, im_placeholder)

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(params['learning_rate'], global_step,
                                                   1, 0.99997592083, staircase=True)

        train_op = training(_loss, learning_rate, global_step)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        # Create a saver for writing training checkpoints.
        sess  = tf.Session()
        sess.run(init)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        saver = tf.train.Saver()

        # Start the training loop.
        for step in xrange(1, params['max_iter'] + 1):
            start_time = time.time()

            feed_dict = fill_feed_dict(params['batch_size'], im_placeholder, noiseim_placeholder)

            _, loss_value = sess.run([train_op, _loss], feed_dict=feed_dict)
            duration = time.time() - start_time

            # save checkpoints
            if step % params['checkpoint'] == 0:
                checkpoint_file = os.path.join(log_dir, 'INPATINGCNN-' + str(step) + ".ckpt")
                saver.save(sess, checkpoint_file)
                print('saved checkpoint on iter %d to %s' % (step, checkpoint_file))

            if step % params['log_print'] == 0:
                print('training loss = %.4f on iter %d (%.6f lr, %.3f sec)' % (loss_value, step, learning_rate.eval(session = sess), duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

def main(params):

    global train_gray
    if params["train_gray"] == 1:
        train_gray = True
    else:
        train_gray = False

    if not train_gray:
        log_dir = "./color-model/"
    else:
        log_dir = "./gray-model/"

    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
    else:
        tf.gfile.DeleteRecursively(log_dir)
    train(params, log_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=64, type=int, help='training batch size')

    parser.add_argument('--max_iter', default=100000, type=int, help='max train iteration')
    parser.add_argument('--train_gray', default=0, type=int, help='1=train gray model, 0=train color model')
    parser.add_argument('--learning_rate', default=3e-3, type=float, help='basic learning rate')

    parser.add_argument('--log_print', default=500, type=int, help='interval between print interval')
    parser.add_argument('--checkpoint', default=5000, type=int, help='interval between save checkpoint')


    args = parser.parse_args()
    params = vars(args)
    print json.dumps(params, indent=2)

    main(params)
