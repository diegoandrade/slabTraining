
# -*- coding: utf-8 -*-

# generate new kinds of slabs

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import numpy as np
import cv2
import random
import scipy.misc
from utils import *
import time
from fidi import *

import imageio



import glob
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'





slim = tf.contrib.slim


HEIGHT, WIDTH, CHANNEL = 64, 64, 1
BATCH_SIZE = 10
EPOCH = 50000
version = 'newSlab'
newSlab_path = './' + version
newDataClahe = './dataClahe'

def lrelu(x, n, leak=0.2):
    return tf.maximum(x, leak * x, name=n)

def process_data():
    current_dir = os.getcwd()
    # parent = os.path.dirname(current_dir)
    slab_dir = os.path.join(current_dir, 'data')
    images = []
    for each in os.listdir(slab_dir):
        images.append(os.path.join(slab_dir,each))
    # print images
    all_images = tf.convert_to_tensor(images, dtype = tf.string)

    images_queue = tf.train.slice_input_producer(
                                        [all_images])

    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels = CHANNEL)
    # sess1 = tf.Session()
    # print sess1.run(image)
    image = tf.image.random_flip_left_right(image) ## why am i doing this here ?? dfa
    image = tf.image.random_brightness(image, max_delta = 0.1)
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
    # noise = tf.Variable(tf.truncated_normal(shape = [HEIGHT,WIDTH,CHANNEL], dtype = tf.float32, stddev = 1e-3, name = 'noise'))
    # print image.get_shape()
    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    image.set_shape([HEIGHT,WIDTH,CHANNEL])
    # image = image + noise
    # image = tf.transpose(image, perm=[2, 0, 1])
    # print image.get_shape()

    image = tf.cast(image, tf.float32)
    image = image / 255.0

    images_batch = tf.train.shuffle_batch(
                                    [image],
                                    batch_size = BATCH_SIZE,
                                    num_threads = 4,
                                    capacity = 200 + 3*BATCH_SIZE,
                                    min_after_dequeue = 20)

    num_images = len(images)

    print(num_images)

    return images_batch, num_images

def generator(input, random_dim, is_train, reuse=False):
    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32 # channel num
    s4 = 4
    output_dim = CHANNEL  # RGB image
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=[random_dim, s4 * s4 * c4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')
         #Convolution, bias, activation, repeat!
        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')
        # 8*8*256
        #Convolution, bias, activation, repeat!
        conv2 = tf.layers.conv2d_transpose(act1, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = tf.nn.relu(bn2, name='act2')
        # 16*16*128
        conv3 = tf.layers.conv2d_transpose(act2, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = tf.nn.relu(bn3, name='act3')
        # 32*32*64
        conv4 = tf.layers.conv2d_transpose(act3, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = tf.nn.relu(bn4, name='act4')
        # 64*64*32
        conv5 = tf.layers.conv2d_transpose(act4, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv5')
        #bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
        act5 = tf.nn.tanh(conv5, name='act5')

        # #128*128*3
        # conv6 = tf.layers.conv2d_transpose(act5, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
        #                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        #                                    name='conv6')
        # # bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn6')
        # act6 = tf.nn.tanh(conv6, name='act6')
        return act5


def discriminator(input, is_train, reuse=False):
    c2, c4, c8, c16 = 64, 128, 256, 512  # channel num: 64, 128, 256, 512
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()

        #Convolution, activation, bias, repeat!
        conv1 = tf.layers.conv2d(input, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
        act1 = lrelu(conv1, n='act1')
         #Convolution, activation, bias, repeat!
        conv2 = tf.layers.conv2d(act1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = lrelu(bn2, n='act2')
        #Convolution, activation, bias, repeat!
        conv3 = tf.layers.conv2d(act2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = lrelu(bn3, n='act3')
         #Convolution, activation, bias, repeat!
        conv4 = tf.layers.conv2d(act3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = lrelu(bn4, n='act4')

        # start from act4
        dim = int(np.prod(act4.get_shape()[1:]))
        print("dim: %d" % dim)
        fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')


        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        # wgan just get rid of the sigmoid
        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        # dcgan
        #acted_out = tf.nn.sigmoid(logits)
        return logits #, acted_out


def train():
    random_dim = 100

    with tf.variable_scope('input'):
        #real and fake image placholders
        real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')

    # wgan
    fake_image = generator(random_input, random_dim, is_train)
    print(".....fake_image pass")
    real_result = discriminator(real_image, is_train)
    print(".....real_pass pass")
    fake_result = discriminator(fake_image, is_train, reuse=True)
    print(".....fake_res pass")

    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  # This optimizes the discriminator.
    g_loss = -tf.reduce_mean(fake_result)  # This optimizes the generator.

    #d_loss = tf.reduce_mean(tf.log(real_result)) - tf.log(1 - fake_result)  # This optimizes the discriminator.
    #g_loss = -tf.reduce_mean(tf.log(fake_result))  # This optimizes the generator.


    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(g_loss, var_list=g_vars)
    # clip discriminator weights
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]


    batch_size = BATCH_SIZE
    image_batch, samples_num = process_data()

    batch_num = int(samples_num / batch_size)
    #total_batch = 0
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # continue training
    save_path = saver.save(sess, "/tmp/model.ckpt")
    ckpt = tf.train.latest_checkpoint('./model/' + version)
    saver.restore(sess, ckpt)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('total training sample num:%d' % samples_num)
    print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, EPOCH))
    print('start training...')
    start_time = time.time()

    passDLoss = -1000
    passGloss = 1000

    for i in range(EPOCH):
        epoch_time = time.time()
        text_file = open("Output.txt", "a+")
        text_file_p = open("Performance.txt", "a+")

        print("Running epoch {}/{}...".format(i, EPOCH))
        for j in range(batch_num):
            print(j)
            d_iters = 5
            g_iters = 1

            train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            for k in range(d_iters):
                print(k)
                train_image = sess.run(image_batch)
                #wgan clip weights
                sess.run(d_clip)

                # Update the discriminator
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={random_input: train_noise, real_image: train_image, is_train: True})

            # Update the generator
            for k in range(g_iters):
                # train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
                _, gLoss = sess.run([trainer_g, g_loss],
                                    feed_dict={random_input: train_noise, is_train: True})

            print ('train:[%d/%d],d_loss:%f,g_loss:%f' % (i, j, dLoss, gLoss))


        #save check point in the best trainig point
        #if (dLoss > passDLoss) and ( dLoss< 0 ) and (gLoss < passGloss) and ( gLoss> 0 ):
        if (gLoss < passGloss) and ( gLoss> 0 ):
            saver.save(sess, './model/' +version + '/' + str(i))
            passDLoss = dLoss
            passGloss = gLoss
            print('g_loss:%f d_loss:%f @iteration:%d' % (dLoss,gLoss,i))
            text_file_p.write("%d \t\t%f \t\t%f \n" % (i, dLoss, gLoss))
            text_file_p.close()

        # save check point every 500 epoch
        if i%100 == 0:
            if not os.path.exists('./model/' + version):
                os.makedirs('./model/' + version)
            saver.save(sess, './model/' +version + '/' + str(i))


        if i%100 == 0:
            # save images
            if not os.path.exists(newSlab_path):
                os.makedirs(newSlab_path)
            sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})
            # imgtest = imgtest * 255.0
            # imgtest.astype(np.uint8)
            save_images(imgtest, [8,8] ,newSlab_path + '/epoch' + str(i) + '.jpg')

        if i%10 == 0:
            print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))
            print("--- total time : %s seconds ---" % (time.time() - start_time))
            text_file.write("%d \t\t%f \t\t%f \t\t%s \t\t%s \n" % (i, dLoss, gLoss, (time.time() - epoch_time),(time.time() - start_time)))
            text_file.close()

    #text_file.close()
    coord.request_stop()
    coord.join(threads)


def test():
    random_dim = 100
    with tf.variable_scope('input'):
        real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')


    # wgan
    fake_image = generator(random_input, random_dim, is_train)
    real_result = discriminator(real_image, is_train)
    fake_result = discriminator(fake_image, is_train, reuse=True)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    variables_to_restore = slim.get_variables_to_restore(include=['gen'])
    #print(variables_to_restore)
    saver = tf.train.Saver(variables_to_restore)
    ckpt = tf.train.latest_checkpoint('./model/' + version)
    saver.restore(sess, ckpt)

    batch_size = BATCH_SIZE
    sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
    imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})



    # save_images(imgtest, [7,7],newSlab_path + '/test1.jpg') #merge_images
    #
    # imsave_solo(imgtest[0], newSlab_path + '/test3.jpg') #imsave_solo images

    for k in range(batch_size):
        save_images_solo(imgtest[k], (newSlab_path + '/sol/longitudinalCrack%d.jpg') % k)


    #save_images(imgtest, [1,1],newSlab_path + '/test1.jpg')
def utilClahe():
    current_dir = os.getcwd()
    # parent = os.path.dirname(current_dir)
    slab_dir = os.path.join(current_dir, 'data')
    images = []
    count = 0
    for each in os.listdir(slab_dir):
        images.append(os.path.join(slab_dir,each))
        count+=1
    print(count)
    #Used to equalize the images currently used
    equ=None
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))

    for k in range(count-1):
        print(k)
        img = cv2.imread(images[k+1], cv2.IMREAD_GRAYSCALE)
        equ=cv2.normalize(img, equ, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imwrite((newDataClahe + '/%d_n.png') % k, equ)
        hist = cv2.equalizeHist(img)
        cv2.imwrite((newDataClahe + '/%d_e.png') % k, hist)
        cl = clahe.apply(img)
        cv2.imwrite((newDataClahe + '/%d_c.png') % k, cl)


def fidDistance():
    # equ1=None
    # equ2=None
    # current_dir = os.getcwd()
    # # parent = os.path.dirname(current_dir)
    # slab_dir = os.path.join(current_dir, 'FID')
    # img1 = cv2.imread("false.png", cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread("real.png")
    # equ1=cv2.normalize(img1, equ2, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    # equ2=cv2.normalize(img2, equ1, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    # im1 = np.ndarray(img1)
    # im2 = np.ndarray(img2)
    #
    # im3 = im1.reshape([1, 3, 64, 64])
    # im4 = im2.reshape([1, 3, 64, 64])
    #
    # # im1 = imageio.imread("FID/false.png")
    # # im2 = imageio.imread("FID/false.png")
    # # print(im2.shape)
    #
    #
    # # im.shape
    # # print type(im)
    # get_fid(im1, im2)

    current_dir = os.getcwd()

    # Paths
    image_path = os.path.join(current_dir, 'FID')
    stats_path = os.path.join(current_dir+'fid_stats_cifar10.npz') # training set statistics
    inception_path = fidi.check_or_download_inception(current_dir) # download inception network

    # loads all images into memory (this might require a lot of RAM!)
    image_list = glob.glob(os.path.join(datapath, '*.png'))
    images = np.array([imread(str(fn)).astype(np.float32) for fn in files])

    # load precalculated training set statistics
    f = np.load(path)
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]
    f.close()

    fidi.create_inception_graph(inception_path)  # load the graph into the current TF graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu_gen, sigma_gen = fidi.calculate_activation_statistics(images, sess, batch_size=100)

    fid_value = fidi.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    print("FID: %s" % fid_value)





if __name__ == "__main__":
    #train()
    #test()
    #utilClahe()
    #restoreChkPoint()
    fidDistance()
