"""
Training script for partial scan completion with a multi-scale conditional
adversarial network. Everything is in this file.

To train from scratch, directories at the start of the script for training
data and logging will need to be changed to your directories.

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse

import numpy as np
import tensorflow as tf

import cv2
from scipy.misc import imread

from scipy import ndimage as nd

import time

import os, random
from urllib.request import urlopen

from PIL import Image
from PIL import ImageDraw

import functools
import itertools

import collections
import six

from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training import device_setter
from tensorflow.contrib.learn.python.learn import run_config

from tensorflow.contrib.framework.python.ops import add_arg_scope

slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.DEBUG)

scale = 0 #Make scale large to spped up initial testing

gen_features0 = 32 if not scale else 1
gen_features1 = 64 if not scale else 1
gen_features2 = 64 if not scale else 1
gen_features3 = 32 if not scale else 1

nin_features1 = 128 if not scale else 1
nin_features2 = 256 if not scale else 1
nin_features3 = 512 if not scale else 1
nin_features4 = 768 if not scale else 1

features1 = 64 if not scale else 1
features2 = 128 if not scale else 1
features3 = 256 if not scale else 1
features4 = 512 if not scale else 1
features5 = features4 if not scale else 1

num_global_enhancer_blocks = 6
num_local_enhancer_blocks = 3

data_dir = "//Desktop-sa1evjv/f/ARM_scans-crops/"

modelSavePeriod = 8 #Train timestep in hours
modelSavePeriod *= 3600 #Convert to s
model_dir = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/stem-random-walk-nin-20-48/"

shuffle_buffer_size = 5000
num_parallel_calls = 6
num_parallel_readers = 6
prefetch_buffer_size = 12
batch_size = 1
num_gpus = 1

#batch_size = 8 #Batch size to use during training
num_epochs = 10**10 #Dataset repeats indefinitely

logDir = "C:/dump/train/"
log_file = model_dir+"log.txt"
val_log_file = model_dir+"val_log.txt"
discr_pred_file = model_dir+"discr_pred.txt"
log_every = 1 #Log every _ examples
cumProbs = np.array([]) #Indices of the distribution plus 1 will be correspond to means

numMeans = 64 // batch_size
scaleMean = 4 #Each means array index increment corresponds to this increase in the mean
numDynamicGrad = 1 #Number of gradients to calculate for each possible mean when dynamically updating training
lossSmoothingBoxcarSize = 5

channels = 1 #Greyscale input image

#Sidelength of images to feed the neural network
cropsize = 512
use_mask = True #If true, supply mask to network as additional information
generator_input_size = cropsize
height_crop = width_crop = cropsize



def generator_architecture(inputs, small_inputs, mask, small_mask, norm_decay, init_pass):
    """Generates data that looks real to discriminators"""

    with tf.variable_scope("Network", reuse=not init_pass):

        def gaussian_noise(x, sigma=0.3, deterministic=False, name=''):
            with tf.variable_scope(name):
                if deterministic:
                    return x
                else:
                    noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32)
                    return x + noise

        concat_axis = 3

        def int_shape(x):
            return list(map(int, x.get_shape()))

        def _actv_func(x):
            x = tf.nn.relu(x)
            return x
        
        def get_var_maybe_avg(var_name, ema, **kwargs):
            ''' utility for retrieving polyak averaged params '''
            v = tf.get_variable(var_name, **kwargs)
            if ema is not None:
                v = ema.average(v)
            return v

        def get_vars_maybe_avg(var_names, ema, **kwargs):
            ''' utility for retrieving polyak averaged params '''
            vars = []
            for vn in var_names:
                vars.append(get_var_maybe_avg(vn, ema, **kwargs))
            return vars

        def mean_only_batch_norm_impl(x, pop_mean, b, is_conv_out=True, deterministic=False, 
                                      decay=norm_decay, name='meanOnlyBatchNormalization'):
            '''
            input comes in which is t=(g*V/||V||)*x
            deterministic : separates training and testing phases
            '''

            with tf.variable_scope(name):
                if deterministic:
                    # testing phase, return the result with the accumulated batch mean
                    return x - pop_mean + b
                else:
                    # compute the current minibatch mean
                    if is_conv_out:
                        # using convolutional layer as input
                        m, _ = tf.nn.moments(x, [0,1,2])
                    else:
                        # using fully connected layer as input
                        m, _ = tf.nn.moments(x, [0])

                    # update minibatch mean variable
                    pop_mean_op = tf.assign(pop_mean, tf.scalar_mul(0.99, pop_mean) + tf.scalar_mul(1-0.99, m))
                    with tf.control_dependencies([pop_mean_op]):
                        return x - m + b

        def batch_norm_impl(x,is_conv_out=True, deterministic=False, decay=norm_decay, name='BatchNormalization'):
            with tf.variable_scope(name):
                scale = tf.get_variable('scale',shape=x.get_shape()[-1],
                                        dtype=tf.float32,initializer=tf.ones_initializer(),trainable=True)
                beta = tf.get_variable('beta',shape=x.get_shape()[-1],
                                       dtype=tf.float32,initializer=tf.zeros_initializer(),trainable=True)
                pop_mean = tf.get_variable('pop_mean',shape=x.get_shape()[-1],
                                           dtype=tf.float32,initializer=tf.zeros_initializer(), trainable=False)
                pop_var = tf.get_variable('pop_var',shape=x.get_shape()[-1],
                                          dtype=tf.float32,initializer=tf.ones_initializer(), trainable=False)
        
                if deterministic:
                    return tf.nn.batch_normalization(x,pop_mean,pop_var,beta,scale,0.001)
                else:
                    if is_conv_out:
                        batch_mean, batch_var = tf.nn.moments(x,[0,1,2])
                    else:
                        batch_mean, batch_var = tf.nn.moments(x,[0])
            
                    pop_mean_op = tf.assign(pop_mean, pop_mean * 0.99 + batch_mean * (1 - 0.99))
                    pop_var_op = tf.assign(pop_var, pop_var * 0.99 + batch_var * (1 - 0.99))
            
                    with tf.control_dependencies([pop_mean_op, pop_var_op]):
                        return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, 0.001)

        conv2d_counter = 0
        def conv2d(x, num_filters, stride=1, filter_size=3,  pad='SAME', nonlinearity=_actv_func, init_scale=1., init=init_pass, 
                    use_weight_normalization=True, use_batch_normalization=False, mean_only_norm=True,
                    deterministic=False, slope=0.01):

            filter_size = [filter_size,filter_size]
            stride = [stride,stride]
                
            '''
            deterministic : used for batch normalizations (separates the training and testing phases)
            '''
            nonlocal conv2d_counter
            conv2d_counter += 1
            name = 'conv'+str(conv2d_counter)

            with tf.variable_scope(name):
                V = tf.get_variable('V', shape=filter_size+[int(x.get_shape()[-1]),num_filters], dtype=tf.float32,
                                          initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        
        
                if use_batch_normalization is False: # not using bias term when doing batch normalization, avoid indefinit growing of the bias, according to BN2015 paper
                    b = tf.get_variable('b', shape=[num_filters], dtype=tf.float32,
                                            initializer=tf.constant_initializer(0.), trainable=True)
        
                if mean_only_norm:
                    pop_mean = tf.get_variable('meanOnlyBatchNormalization/pop_mean',shape=[num_filters], 
                                               dtype=tf.float32, initializer=tf.zeros_initializer(),trainable=False)
        
                if use_weight_normalization:
                    g = tf.get_variable('g', shape=[num_filters], dtype=tf.float32,
                                          initializer=tf.constant_initializer(1.), trainable=True)
            
                    if init:
                        v_norm = tf.nn.l2_normalize(V,[0,1,2])
                        x = tf.nn.conv2d(x, v_norm, strides=[1] + stride + [1],padding=pad)
                        m_init, v_init = tf.nn.moments(x, [0,1,2])
                        scale_init=init_scale/tf.sqrt(v_init + 1e-08)
                        g = g.assign(scale_init)
                        b = b.assign(-m_init*scale_init)
                        x = tf.reshape(scale_init,[1,1,1,num_filters])*(x-tf.reshape(m_init,[1,1,1,num_filters]))
                    else:
                        W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])
                        if mean_only_norm: # use weight-normalization combined with mean-only-batch-normalization
                            x = tf.nn.conv2d(x,W,strides=[1]+stride+[1],padding=pad)
                            x = mean_only_batch_norm_impl(x,pop_mean,b,is_conv_out=True, deterministic=deterministic)
                        else:
                            # use just weight-normalization
                            x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1] + stride + [1], pad), b)
                
                elif use_batch_normalization:
                    x = tf.nn.conv2d(x,V,[1]+stride+[1],pad)
                    x = batch_norm_impl(x,is_conv_out=True,deterministic=deterministic)
                else:
                    x = tf.nn.bias_add(tf.nn.conv2d(x,V,strides=[1]+stride+[1],padding=pad),b)
        
                # apply nonlinearity
                if nonlinearity is not None:
                    x = nonlinearity(x, slope)
        
                return x

        deconv2d_counter = 0
        def deconv2d(x, num_filters, stride=1, filter_size=3,  pad='SAME', nonlinearity=_actv_func,
                     init_scale=1., init=init_pass, 
                     use_weight_normalization=True, use_batch_normalization=False, mean_only_norm=True,
                     deterministic=False, name='', slope=0.01):

            filter_size = [filter_size,filter_size]
            stride = [stride,stride]
                
            '''
            deterministic : used for batch normalizations (separates the training and testing phases)
            '''

            nonlocal deconv2d_counter
            deconv2d_counter += 1
            name = 'deconv'+str(deconv2d_counter)

            xs = int_shape(x)
            if pad=='SAME':
                target_shape = [xs[0], xs[1]*stride[0], xs[2]*stride[1], num_filters]
            else:
                target_shape = [xs[0], xs[1]*stride[0] + filter_size[0]-1, xs[2]*stride[1] + filter_size[1]-1, num_filters]
    
            with tf.variable_scope(name):
                V = tf.get_variable('V', shape=filter_size+[num_filters,int(x.get_shape()[-1])], dtype=tf.float32, 
                                    initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
                #V = tf.get_variable('V', shape=filter_size+[int(x.get_shape()[-1]), num_filters], dtype=tf.float32,
                #                    initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        
        
                if use_batch_normalization is False: # not using bias term when doing batch normalization, avoid indefinit growing of the bias, according to BN2015 paper
                    b = tf.get_variable('b', shape=[num_filters], dtype=tf.float32,
                                            initializer=tf.constant_initializer(0.), trainable=True)
        
                if mean_only_norm:
                    pop_mean = tf.get_variable('meanOnlyBatchNormalization/pop_mean',shape=[num_filters], dtype=tf.float32, initializer=tf.zeros_initializer(),trainable=False)
        
                if use_weight_normalization:
                    g = tf.get_variable('g', shape=[num_filters], dtype=tf.float32,
                                        initializer=tf.constant_initializer(1.), trainable=True)
            
                    if init:
                        v_norm = tf.nn.l2_normalize(V,[0,1,2])
                        x = tf.nn.conv2d_transpose(x, v_norm, target_shape, strides=[1] + stride + [1],padding=pad)
                        m_init, v_init = tf.nn.moments(x, [0,1,2])
                        scale_init=init_scale/tf.sqrt(v_init + 1e-08)
                        g = g.assign(scale_init)
                        b = b.assign(-m_init*scale_init)
                        x = tf.reshape(scale_init,[1,1,1,num_filters])*(x-tf.reshape(m_init,[1,1,1,num_filters]))
                    else:
                        W = tf.reshape(g, [1, 1, num_filters, 1]) * tf.nn.l2_normalize(V, [0, 1, 2])
                        if mean_only_norm: # use weight-normalization combined with mean-only-batch-normalization
                            x = tf.nn.conv2d_transpose(x,W,target_shape,strides=[1]+stride+[1],padding=pad)
                            x = mean_only_batch_norm_impl(x,pop_mean,b,is_conv_out=True, deterministic=deterministic)
                        else:
                            # use just weight-normalization
                            x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1] + stride + [1], pad), b)
                
                elif use_batch_normalization:
                    x = tf.nn.conv2d(x,V,[1]+stride+[1],pad)
                    x = batch_norm_impl(x,is_conv_out=True,deterministic=deterministic)
                else:
                    x = tf.nn.bias_add(tf.nn.conv2d(x,V,strides=[1]+stride+[1],padding=pad),b)
        
                # apply nonlinearity
                if nonlinearity is not None:
                    x = nonlinearity(x, slope)
        
                return x

        def xception_middle_block(input, features):
        
            main_flow = conv2d(
                x=input,
                num_filters=features,
                stride=1)
            main_flow = conv2d(
                x=main_flow,
                num_filters=features,
                stride=1)
            main_flow = conv2d(
                x=main_flow,
                num_filters=features,
                stride=1)

            return main_flow + input

        def init_batch_norm(x):
            batch_mean, batch_var = tf.nn.moments(x,[0])
            return (x - batch_mean) / np.sqrt( batch_var + 0.001 )

        def network_in_network(input, nin_features_out, mask=None):

            if use_mask:
                concatenation = tf.concat(values=[input, mask], axis=concat_axis)
            else:
                concatenation = input

            with tf.variable_scope("Inner"):

                nin = conv2d(concatenation, 64, 1,
                             filter_size=5,
                             mean_only_norm=True,
                             use_weight_normalization=not use_mask, slope=0.1)

                nin = conv2d(nin, nin_features1, 2)
                nin = conv2d(nin, nin_features2, 2)
                nin = conv2d(nin, nin_features3, 2)
                nin = conv2d(nin, nin_features4, 2)

                for _ in range(num_global_enhancer_blocks):
                    nin = xception_middle_block(nin, nin_features4)

                nin = deconv2d(nin, nin_features3, 2)
                nin = deconv2d(nin, nin_features2, 2)
                nin = deconv2d(nin, nin_features1, 2)
                nin = deconv2d(nin, nin_features_out, 2)

            return nin

        ##Model building
        if not init_pass:
            input = inputs
            small_input = small_inputs
        else:
            input = tf.random_uniform(shape=int_shape(inputs), minval=-0.8, maxval=0.8)
            input *= mask
            small_input = tf.image.resize_images(input, (cropsize//2, cropsize//2))

        with tf.variable_scope("Inner"):
            if not use_mask:
                nin, inner = network_in_network(small_input, gen_features1)
            else:
                nin, inner = network_in_network(small_input, gen_features1, small_mask)

        with tf.variable_scope("Outer"):

            if use_mask:
                concatenation = tf.concat(values=[input, mask], axis=concat_axis)
            else:
                concatenation = input
            enc = conv2d(x=concatenation,
                         num_filters=gen_features0,
                         stride=1,
                         filter_size=5,
                         mean_only_norm=not use_mask, slope=0.1)

            enc = conv2d(enc, gen_features1, 2, slope=0.1)

            enc = enc + nin

            for _ in range(num_local_enhancer_blocks):
                enc = xception_middle_block(enc, gen_features2)

            enc = deconv2d(enc, gen_features3, 2)
            enc = conv2d(enc, gen_features3, 1)

            outer = conv2d(enc, 1, 1, mean_only_norm=False, nonlinearity=None)

        return outer


def Generator():
    """Load generator ready to complete partial scans."""

    def __init__(self, 
                 ckpt_dir, 
                 generator_architecture=generator_architecture):

        self._partial_scan = tf.placeholder("partial_scan", shape=[None, cropsize, cropsize, channels])
        self._complete_scan = generator_architecture(self._partial_scan)

        with tf.Session(config=sess_config) as sess:

            #Load portion of checkpoint relevant to training
            collection = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, 
                scope="Network")
           
            saver = tf.train.Saver(var_list=collection)
            saver.restore(sess, ckpt_dir)

            #Initialize all variables
            sess.run(tf.global_variables_initializer())

            self._sess = sess

    def infer(self, partial_scan):
        """
        Complete partial scans with deep learning
        
        Inputs:
            partial_scan: Batch of partial scans with shape [batch_size, 512, 512, 1]

        Returns:
            Completed partial scans in the same shape as the inputs.
        """

        feed_dict = { self._partial_scan: partial_scan }

        completed_scans = self._sess.run(
            self._complete_scan,
            feed_dict=feed_dict)

        return completed_scans


def get_example_scan(idx=None):

    if idx == None:
        idx = random.randint(1, 5)

    partial_filepath = os.getcwd().replace("\\", "/") + f"/example_scans/partial_scan-{idx}.tif"
    truth_filepath = os.getcwd().replace("\\", "/") + f"/example_scans/truth-{idx}.tif"

    partial_scan = imread(partial_filepath, mode="F")
    truth = imread(truth_filepath, mode="F")

    return partial_scan, truth


def disp(img):
    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)
    return

if __name__ == "__main__":

    ckpt_dir = "//flexo.ads.warwick.ac.uk/shared41/Jeffrey-Ede/models/stem-random-walk-nin-20-43/notable_ckpts/"

    partial_scan, truth = get_example_scan()

    gen = Generator(ckpt_dir=ckpt_dir)
    disp(gen.infer(partial_scan))