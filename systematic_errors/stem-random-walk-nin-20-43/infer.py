"""
Training script for partial scan completion with a multi-scale conditional
adversarial network. Everything is in this file.

To train from scratch, directories at the start of the script for training
data and logging will need to be changed to your directories.

This is NOT an optimized inference script. It reloads a full session.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

discr_size = 70

coverage = 1/20

disp_select = False #Display selelected pixels upon startup

def int_shape(x):
    return list(map(int, x.get_shape()))

def spectral_norm(w, iteration=1, count=0):
   w0 = w
   w_shape = w.shape.as_list()
   w = tf.reshape(w, [-1, w_shape[-1]])

   u = tf.get_variable("u"+str(count), 
                       [1, w_shape[-1]], 
                       initializer=tf.random_normal_initializer(mean=0.,stddev=0.03), 
                       trainable=False)

   u_hat = u
   v_hat = None
   for i in range(iteration):
       """
       power iteration
       Usually iteration = 1 will be enough
       """
       v_ = tf.matmul(u_hat, tf.transpose(w))
       v_hat = tf.nn.l2_normalize(v_)

       u_ = tf.matmul(v_hat, w)
       u_hat = tf.nn.l2_normalize(u_)

   u_hat = tf.stop_gradient(u_hat)
   v_hat = tf.stop_gradient(v_hat)

   sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

   with tf.control_dependencies([u.assign(u_hat)]):
       w_norm = w / sigma
       w_norm = tf.reshape(w_norm, w_shape)

   return w_norm

adjusted_mse_counter = 0
def adjusted_mse(img1, img2):

    return tf.reduce_mean(tf.abs(img1-img2)) #tf.losses.mean_squared_error(img1, img2)

    def pad(tensor, size):
            d1_pad = size[0]
            d2_pad = size[1]

            paddings = tf.constant([[0, 0], [d1_pad, d1_pad], [d2_pad, d2_pad], [0, 0]], dtype=tf.int32)
            padded = tf.pad(tensor, paddings, mode="REFLECT")
            return padded

    def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                   ):
        """Makes 2D gaussian Kernel for convolution."""

        d = tf.distributions.Normal(mean, std)

        vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

        gauss_kernel = tf.einsum('i,j->ij', vals, vals)

        return gauss_kernel / tf.reduce_sum(gauss_kernel)

    def blur(image):
        gauss_kernel = gaussian_kernel( 1, 0., 1.5 )

        #Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature
        gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]

        #Convolve
        image = pad(image, (1,1))
        return tf.nn.conv2d(image, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")

    #Track average MSEs
    adjusted_mse_counter += 1
    avg = tf.get_variable(
        name=f"avg-{adjusted_mse_counter}",
        shape=img1.get_shape(),
        initializer=3*tf.ones(img1.get_shape()))

    squared_errors = (img1 - img2)**2

    update_avg = tf.assign(avg, 0.999*avg + 0.001*squared_errors)

    with tf.control_dependencies([update_avg]):

        #Errors for px with systematically higher MSEs are increased
        scale = blur(avg)
        scale /= tf.reduce_mean(scale)

        mse = tf.reduce_mean( scale*squared_errors )

    return mse

capper_counter = 0
def capper_fn(x):
    global capper_counter; capper_counter += 1

    mu = tf.get_variable(f"mu-{capper_counter}", initializer=tf.constant(25, dtype=tf.float32))
    mu2 = tf.get_variable(f"mu2-{capper_counter}", initializer=tf.constant(30**2, dtype=tf.float32))

    def cap(x):
        sigma = tf.sqrt(mu2 - mu**2+1.e-8)
        capped_x = tf.cond(x < mu+3*sigma, lambda: x, lambda: x/tf.stop_gradient(x/(mu+3*sigma)))
        return capped_x

    x = cap(x)

    with tf.control_dependencies([mu.assign(0.999*mu+0.001*x), mu2.assign(0.999*mu2+0.001*x**2)]):
        x = cap(x)
        return tf.cond(x <= 1, lambda: x, lambda: tf.sqrt(x + 1.e-8))

def generator_architecture(inputs, small_inputs, mask, small_mask, norm_decay, init_pass):
    """Generates fake data to try and fool the discrimator"""

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

        mu_counter = 0
        def mean_only_batch_norm(input, decay=norm_decay, reuse_counter=None, init=init_pass):
            
            mu = tf.reduce_mean(input, keepdims=True)
            shape = int_shape(mu)

            if not reuse_counter and init_pass: #Variable not being reused
                nonlocal mu_counter
                mu_counter += 1
                running_mean = tf.get_variable("mu"+str(mu_counter), 
                                               dtype=tf.float32, 
                                               initializer=tf.constant(np.zeros(shape, dtype=np.float32)), 
                                               trainable=False)
            else:
                running_mean = tf.get_variable("mu"+str(mu_counter))

            running_mean = decay*running_mean + (1-decay)*mu
            mean_only_norm = input - running_mean

            return mean_only_norm

        def _actv_func(x, slope=0.01):
            #x = tf.nn.leaky_relu(x, slope)
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
            
                    pop_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
                    pop_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            
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

                residuals = False
                if residuals:
                    nin = conv2d(nin, nin_features1, 2, slope=0.1)
                    nin1 = nin
                    nin = conv2d(nin, nin_features2, 2, slope=0.1)
                    nin2 = nin
                    nin = conv2d(nin, nin_features3, 2, slope=0.1)
                    nin3 = nin
                    nin = conv2d(nin, nin_features4, 2, slope=0.1)

                    for _ in range(num_global_enhancer_blocks):
                        nin = xception_middle_block(nin, nin_features4)

                    nin = deconv2d(nin, nin_features3, 2)
                    nin += nin3
                    nin = deconv2d(nin, nin_features2, 2)
                    nin += nin2
                    nin = deconv2d(nin, nin_features1, 2)
                    nin += nin1
                    nin = deconv2d(nin, nin_features_out, 2)
                else:
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
            
            with tf.variable_scope("Trainer"):

                inner = conv2d(nin, 64, 1)
                inner = conv2d(inner, 1, 1, mean_only_norm=False, nonlinearity=None)

            return nin, inner

        ##Model building
        if not init_pass:
            input = inputs
            small_input = small_inputs
        else:
            input = tf.random_uniform(shape=int_shape(inputs), minval=-0.8, maxval=0.8)
            input *= mask
            small_input = tf.image.resize_images(input, (cropsize//2, cropsize//2), align_corners=True)

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

        return inner, outer


def discriminator_architecture(inputs, second_input=None, phase=False, params=None, 
                               gen_loss=0., reuse=False):
    """Three discriminators to discriminate between two data discributions"""

    with tf.variable_scope("GAN/Discr", reuse=reuse):

        def int_shape(x):
            return list(map(int, x.get_shape()))

        #phase = mode == tf.estimator.ModeKeys.TRAIN #phase is true during training
        concat_axis = 3

        def _instance_norm(net, train=phase):
            batch, rows, cols, channels = [i.value for i in net.get_shape()]
            var_shape = [channels]
            mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
            shift = tf.Variable(tf.zeros(var_shape), trainable=False)
            scale = tf.Variable(tf.ones(var_shape), trainable=False)
            epsilon = 1.e-3
            normalized = (net - mu) / (sigma_sq + epsilon)**(.5)
            return scale*normalized + shift

        def instance_then_activ(input):
            batch_then_activ = _instance_norm(input)
            batch_then_activ = tf.nn.relu(batch_then_activ)
            return batch_then_activ

        ##Reusable blocks
        def _batch_norm_fn(input):
            batch_norm = tf.contrib.layers.batch_norm(
                input,
                epsilon=0.001,
                decay=0.999,
                center=True, 
                scale=True,
                is_training=phase,
                fused=True,
                zero_debias_moving_mean=False,
                renorm=False)
            return batch_norm

        def batch_then_activ(input): #Changed to instance norm for stability
            batch_then_activ = input#_instance_norm(input)
            batch_then_activ = tf.nn.leaky_relu(batch_then_activ, alpha=0.2)
            return batch_then_activ

        def conv_block_not_sep(input, filters, kernel_size=3, phase=phase, batch_and_activ=True):
            """
            Convolution -> batch normalisation -> leaky relu
            phase defaults to true, meaning that the network is being trained
            """

            conv_block = slim.conv2d(
                inputs=input,
                num_outputs=filters,
                kernel_size=kernel_size,
                padding="SAME",
                activation_fn=None)

            if batch_and_activ:
                conv_block = batch_then_activ(conv_block)

            return conv_block

        def conv_block(input, filters, phase=phase):
            """
            Convolution -> batch normalisation -> leaky relu
            phase defaults to true, meaning that the network is being trained
            """

            conv_block = strided_conv_block(input, filters, 1, 1)

            return conv_block

        count = 0
        def discr_conv_block(input, filters, stride, rate=1, phase=phase, kernel_size=3, actv=True):
        
            nonlocal count 
            count += 1
            w = tf.get_variable("kernel"+str(count), shape=[kernel_size, kernel_size, input.get_shape()[-1], filters])
            b = tf.get_variable("bias"+str(count), [filters], initializer=tf.constant_initializer(0.0))

            x = tf.nn.conv2d(input=input, filter=spectral_norm(w, count=count), 
                             strides=[1, stride, stride, 1], padding='VALID') + b

            if actv:
                x = batch_then_activ(x)

            return x

        def residual_conv(input, filters):

            residual = slim.conv2d(
                inputs=input,
                num_outputs=filters,
                kernel_size=1,
                stride=2,
                padding="SAME",
                activation_fn=None)
            residual = batch_then_activ(residual)

            return residual

        def xception_encoding_block(input, features):
        
            cnn = conv_block(
                input=input, 
                filters=features)
            cnn = conv_block(
                input=cnn, 
                filters=features)
            cnn = strided_conv_block(
                input=cnn,
                filters=features,
                stride=2)

            residual = residual_conv(input, features)
            cnn += residual

            return cnn

        def xception_encoding_block_diff(input, features_start, features_end):
        
            cnn = conv_block(
                input=input, 
                filters=features_start)
            cnn = conv_block(
                input=cnn, 
                filters=features_start)
            cnn = strided_conv_block(
                input=cnn,
                filters=features_end,
                stride=2)

            residual = residual_conv(input, features_end)
            cnn += residual

            return cnn

        def xception_middle_block(input, features):
        
            main_flow = strided_conv_block(
                input=input,
                filters=features,
                stride=1)
            main_flow = strided_conv_block(
                input=main_flow,
                filters=features,
                stride=1)
            main_flow = strided_conv_block(
                input=main_flow,
                filters=features,
                stride=1)

            return main_flow + input

        def shared_flow(input, layers):

            shared = xception_encoding_block_diff(input, features2, features3)
            layers.append(shared)
            shared = xception_encoding_block_diff(shared, features3, features4)
            layers.append(shared)
        
            shared = xception_encoding_block(shared, features5)
            layers.append(shared)

            shared = xception_middle_block(shared, features5)
            layers.append(shared)
            shared = xception_middle_block(shared, features5)
            layers.append(shared)
            shared = xception_middle_block(shared, features5)
            layers.append(shared)
            shared = xception_middle_block(shared, features5)
            layers.append(shared)

            return shared, layers

        def terminating_fc(input):

            fc = tf.reduce_mean(input, [1,2])
            fc = tf.reshape(fc, (-1, features5))
            fc = tf.contrib.layers.fully_connected(inputs=fc,
                                                   num_outputs=1,
                                                   activation_fn=None)
            return fc

        def max_pool(input, size=2, stride=2):

            pool = tf.contrib.layers.max_pool2d(inputs=input,
                                                kernel_size=size,
                                                stride=stride,
                                                padding='SAME')
            return pool

        
        testing_scale = 1
        features1 = 64 // testing_scale
        features2 = 128 // testing_scale
        features3 = 256 // testing_scale
        features4 = 512 // testing_scale

        def discriminate(x):
            """Discriminator architecture"""

            x = discr_conv_block(x, features1, 2, 1, kernel_size=4)
            x = discr_conv_block(x, features2, 2, 1, kernel_size=4)
            x = discr_conv_block(x, features3, 2, 1, kernel_size=4)
            #x = discr_conv_block(x, features3, 1, 1, kernel_size=4)
            x = discr_conv_block(x, features4, 2, 1, kernel_size=4)
            x = tf.reduce_sum(x, axis=[1,2,3])
            #shape = int_shape(x)
            #x = tf.reshape(x, (-1, shape[1]*shape[2]*shape[3]))
            #x = tf.contrib.layers.fully_connected(
            #    inputs=x, num_outputs=1, biases_initializer=None, activation_fn=None)

            return x


        '''Model building'''        
        with tf.variable_scope("small", reuse=reuse) as small_scope:
            small = inputs[0]
            small = discriminate(small)

        with tf.variable_scope("medium", reuse=reuse) as medium_scope:
            medium = inputs[1]
            medium = discriminate(medium)

        with tf.variable_scope("large", reuse=reuse) as large_scope:
            large = inputs[2]
            large = discriminate(large)


    discriminations = []
    for x in [small, medium, large]:
        clipped = x#tf.clip_by_value(x, clip_value_min=0, clip_value_max=1000) #5*l2_norm
        discriminations.append( clipped )

    return discriminations


def experiment(feature, ground_truth, mask, learning_rate_ph, discr_lr_ph, beta1_ph, 
               discr_beta1_ph, norm_decay, train_outer_ph, ramp_ph, initialize):

    def pad(tensor, size):
            d1_pad = size[0]
            d2_pad = size[1]

            paddings = tf.constant([[0, 0], [d1_pad, d1_pad], [d2_pad, d2_pad], [0, 0]], dtype=tf.int32)
            padded = tf.pad(tensor, paddings, mode="REFLECT")
            return padded

    def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                   ):
        """Makes 2D gaussian Kernel for convolution."""

        d = tf.distributions.Normal(mean, std)

        vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

        gauss_kernel = tf.einsum('i,j->ij', vals, vals)

        return gauss_kernel / tf.reduce_sum(gauss_kernel)

    def blur(image):
        gauss_kernel = gaussian_kernel( 2, 0., 2.5 )

        #Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature
        gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]

        #Convolve
        image = pad(image, (2,2))
        return tf.nn.conv2d(image, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")

    def get_multiscale_crops(input, multiscale_channels=1):
        """Assumes square inputs"""

        input = pad(input, (2*discr_size, 2*discr_size)) #Extra padding to reduce periodic artefacts

        s = int_shape(input)

        small = tf.random_crop(
                    input,
                    size=(batch_size, discr_size, discr_size, multiscale_channels))
        small = tf.image.resize_images(small, (discr_size, discr_size), align_corners=True)
        medium = tf.random_crop(
                    input,
                    size=(batch_size, 2*discr_size, 2*discr_size, multiscale_channels))
        medium = tf.image.resize_images(medium, (discr_size, discr_size), align_corners=True)
        large = tf.random_crop(
                    input,
                    size=(batch_size, 4*discr_size, 4*discr_size, multiscale_channels))
        large = tf.image.resize_images(large, (discr_size, discr_size), align_corners=True)

        return small, medium, large

    #Generator
    feature = tf.reshape(feature, [-1, cropsize, cropsize, channels])
    feature_small = tf.image.resize_images(feature, (cropsize//2, cropsize//2), align_corners=True)
    truth = tf.reshape(ground_truth, [-1, cropsize, cropsize, channels])
    truth_small = tf.image.resize_images(truth, (cropsize//2, cropsize//2), align_corners=True)
    small_mask = tf.image.resize_images(mask, (cropsize//2, cropsize//2), align_corners=True)

    if initialize:
        print("Started initialization")
        _, _ = generator_architecture(
            feature,  feature_small, mask, small_mask, norm_decay, init_pass=True)
    print("Initialized")
    output_inner, output_outer = generator_architecture(
        feature, feature_small, mask, small_mask, norm_decay, init_pass=False)
    print("Architecture ready")

    #Blurred images
    blur_truth_small = blur(truth_small)
    blur_output_inner = blur(output_inner)
    blur_truth = blur(truth)
    blur_output_outer = blur(output_outer)

    #Trainable parameters
    model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Network")

    model_params_inner = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Network/Inner/Inner")
    model_params_trainer = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Network/Inner/Trainer")

    model_params_outer = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Network/Outer")

    ##Discriminators
    #Intermediate image for gradient penalty calculation
    epsilon = tf.random_uniform(
                shape=[2, 1, 1, 1, 1],
				minval=0.,
				maxval=1.)
    X_hat_outer = (1-epsilon[0])*truth + epsilon[0]*output_outer
    X_hat_inner = (1-epsilon[1])*blur_truth_small + epsilon[1]*output_inner

    discr_inputs_outer = [output_outer, truth, X_hat_outer]
    discr_inputs_inner = [output_inner, blur_truth_small, X_hat_inner]

    #Crop images at multiple scales at the same places for each scale
    concat_outer = tf.concat(discr_inputs_outer, axis=3)
    concat_inner = tf.concat(discr_inputs_inner, axis=3)
    num_channels_outer = len(discr_inputs_outer)
    num_channels_inner = len(discr_inputs_inner)
    multiscale_crops_outer = get_multiscale_crops(concat_outer, multiscale_channels=num_channels_outer)
    multiscale_crops_inner = get_multiscale_crops(concat_inner, multiscale_channels=num_channels_inner)
    multiscale_crops_outer = [tf.unstack(crop, axis=3) for crop in multiscale_crops_outer]
    multiscale_crops_inner = [tf.unstack(crop, axis=3) for crop in multiscale_crops_inner]

    #Sort crops into categories
    shape = (batch_size, discr_size, discr_size, channels)
    crops_set_outer = []
    for crops in multiscale_crops_outer:
        crops_set_outer.append( [tf.reshape(unstacked, shape) for unstacked in crops] )
    crops_set_inner = []
    for crops in multiscale_crops_inner:
        crops_set_inner.append( [tf.reshape(unstacked, shape) for unstacked in crops] )

    #Get intermediate representations
    multiscale_xhat_outer = [m[2] for m in crops_set_outer]
    multiscale_xhat_inner = [m[2] for m in crops_set_inner]

    #Concatenate so the crops can be processed as a single batch
    multiscale_outer = []
    for crops in crops_set_outer:
        multiscale_outer.append( tf.concat(crops, axis=0) )
    multiscale_inner = []
    for crops in crops_set_inner:
        multiscale_inner.append( tf.concat(crops, axis=0) )

    _discrimination_outer = discriminator_architecture( multiscale_outer )
    _discrimination_inner = discriminator_architecture( multiscale_inner, reuse=True )

    model_params_discr_small = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Discr/small")
    model_params_discr_medium = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Discr/medium")
    model_params_discr_large = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Discr/large")

    model_params_discrs = [model_params_discr_small,
                           model_params_discr_medium,
                           model_params_discr_large]

    #Separate batch into discrimination categories
    discr_of_output_outer = [d[0] for d in _discrimination_outer]
    discr_of_truth = [d[1] for d in _discrimination_outer]
    discr_of_X_hat_outer = [d[2] for d in _discrimination_outer]

    discr_of_output_inner = [d[0] for d in _discrimination_inner]
    discr_of_truth_small = [d[1] for d in _discrimination_inner]
    discr_of_X_hat_inner = [d[2] for d in _discrimination_inner]

    pred_real_outer = 0.
    pred_fake_outer = 0.
    avg_d_grads_outer = 0.
    d_losses_outer = []

    pred_real_inner = 0.
    pred_fake_inner = 0.
    avg_d_grads_inner = 0.
    d_losses_inner = []

    wass_weight = 1.
    gradient_penalty_weight = 10.
    l2_inner_weight = 5.e-5
    l2_outer_weight = 5.e-5

    def get_gradient_penalty(_discr_of_X_hat, _multiscale_xhat):

        grad_D_X_hat = tf.gradients(_discr_of_X_hat, [_multiscale_xhat])[0]
        red_idx = [i for i in range(2, _multiscale_xhat.shape.ndims)]
        slopes = tf.sqrt(1.e-8+tf.reduce_sum(tf.square(grad_D_X_hat), axis=red_idx))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        return gradient_penalty

    ##Losses and train ops
    wass_loss_for_gen_outer = 0.
    wass_loss_for_gen_inner = 0.
    wass_loss_for_discr_outer = 0.
    wass_loss_for_discr_inner = 0.
    for i in range(3): #Discrimination is on 3 scales

        #wasserstein_loss_outer = discr_of_output_outer[i] - discr_of_truth[i]
        #wasserstein_loss_inner = discr_of_output_inner[i] - discr_of_truth_small[i]

        #wass_loss_for_discr_outer += wasserstein_loss_outer
        #wass_loss_for_discr_inner += wasserstein_loss_inner

        #wass_loss_for_gen_outer += -discr_of_output_outer[i]
        #wass_loss_for_gen_inner += -discr_of_output_inner[i]

        gradient_penalty_outer = 0.#get_gradient_penalty(discr_of_X_hat_outer[i], multiscale_xhat_outer[i])
        gradient_penalty_inner = 0.#get_gradient_penalty(discr_of_X_hat_inner[i], multiscale_xhat_inner[i])

        wasserstein_loss_outer = tf.pow(discr_of_truth[i]-1., 2) + tf.pow(discr_of_output_outer[i], 2)
        wasserstein_loss_inner = tf.pow(discr_of_truth_small[i]-1., 2) + tf.pow(discr_of_output_inner[i], 2)

        wass_loss_for_discr_outer += wasserstein_loss_outer
        wass_loss_for_discr_inner += wasserstein_loss_inner

        wass_loss_for_gen_outer += tf.pow(discr_of_output_outer[i]-1., 2)
        wass_loss_for_gen_inner += tf.pow(discr_of_output_inner[i]-1, 2)

        pred_real_outer += discr_of_truth[i]
        pred_fake_outer += discr_of_output_outer[i]
        avg_d_grads_outer += gradient_penalty_outer
        pred_real_inner += discr_of_truth_small[i]
        pred_fake_inner += discr_of_output_inner[i]
        avg_d_grads_inner += gradient_penalty_inner

        d_loss_outer = wass_weight*wasserstein_loss_outer + gradient_penalty_weight*gradient_penalty_outer
        d_loss_inner = wass_weight*wasserstein_loss_inner + gradient_penalty_weight*gradient_penalty_inner

        d_losses_outer.append(d_loss_outer)
        d_losses_inner.append(d_loss_inner)

    mse_inner = np.sqrt(200)*adjusted_mse(blur_truth_small, output_inner)
    mse_inner = capper_fn(mse_inner)
    #mse_inner = 2.*tf.cond( mse_inner < 1, lambda: mse_inner, lambda: tf.sqrt(mse_inner+1.e-8) )
    #mse_inner = tf.minimum(mse_inner, 50)

    mse_outer = np.sqrt(200)*adjusted_mse(blur_truth, output_outer)
    mse_outer = capper_fn(mse_outer)
    #mse_outer = 2.*tf.cond( mse_outer < 1, lambda: mse_outer, lambda: tf.sqrt(mse_outer+1.e-8) )
    #mse_outer = tf.minimum(mse_outer, 50) #Safegaurd against error spikes

    mse_outer_together = np.sqrt(200)*adjusted_mse(blur_truth, blur_output_outer)
    mse_outer_together = capper_fn(mse_outer_together)
    #mse_outer_together = 2.*tf.cond( mse_outer < 1, lambda: mse_outer, lambda: tf.sqrt(mse_outer+1.e-8) )

    #mse_inner = 10*tf.reduce_mean(tf.abs( blur_truth_small - blur_output_inner ))
    #mse_outer = 10*tf.reduce_mean(tf.abs( blur_truth - blur_output_outer ))

    loss = mse_outer_together + 2*wass_loss_for_gen_outer
    loss_inner = mse_inner 
    loss_outer = mse_outer

    train_ops_discr = []
    for i in range(3):
        d_loss = tf.cond( train_outer_ph, lambda: d_losses_outer[i], lambda: d_losses_inner[i] )
        d_train_op = tf.train.AdamOptimizer(discr_lr_ph, 0.9).minimize(
            d_loss, var_list=model_params_discrs[i])
        train_ops_discr.append(d_train_op)

    #Provision inner network with an ancillary loss tower 
    train_op_trainer = tf.train.AdamOptimizer(learning_rate_ph, 0.9).minimize(
        2*loss_inner, var_list=model_params_trainer)

    train_op_inner_start = tf.train.AdamOptimizer(learning_rate_ph, 0.9).minimize(
        loss_inner+loss_outer, var_list=model_params_inner)
    train_op_inner_end = tf.train.AdamOptimizer(learning_rate_ph, 0.9).minimize(
        loss_inner+loss, var_list=model_params_inner)

    train_op_outer_start = tf.train.AdamOptimizer(learning_rate_ph, 0.9).minimize(
        loss_outer, var_list=model_params_outer)
    train_op_outer_end = tf.train.AdamOptimizer(learning_rate_ph, 0.9).minimize(
        loss, var_list=model_params_outer)

    start_train_ops = [train_op_inner_start, train_op_outer_start, train_op_trainer]
    end_train_ops = [train_op_inner_end, train_op_outer_end, train_op_trainer]

    return {'start_train_ops': start_train_ops, 
            'end_train_ops': end_train_ops, 
            'train_ops_discr': train_ops_discr,
            'output_inner': output_inner, 
            'output_outer': output_outer, 
            'mse_inner': mse_inner, 
            'mse_outer': mse_outer,
            'wass_loss_inner': wass_loss_for_gen_inner,
            'wass_loss_outer': wass_loss_for_gen_outer,
            'wass_loss_d_inner': wass_loss_for_discr_inner,
            'wass_loss_d_outer': wass_loss_for_discr_outer
            }


class RunConfig(tf.contrib.learn.RunConfig): 
    def uid(self, whitelist=None):
        """
        Generates a 'Unique Identifier' based on all internal fields.
        Caller should use the uid string to check `RunConfig` instance integrity
        in one session use, but should not rely on the implementation details, which
        is subject to change.
        Args:
          whitelist: A list of the string names of the properties uid should not
            include. If `None`, defaults to `_DEFAULT_UID_WHITE_LIST`, which
            includes most properties user allowes to change.
        Returns:
          A uid string.
        """
        if whitelist is None:
            whitelist = run_config._DEFAULT_UID_WHITE_LIST

        state = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        # Pop out the keys in whitelist.
        for k in whitelist:
            state.pop('_' + k, None)

        ordered_state = collections.OrderedDict(
            sorted(state.items(), key=lambda t: t[0]))
        # For class instance without __repr__, some special cares are required.
        # Otherwise, the object address will be used.
        if '_cluster_spec' in ordered_state:
            ordered_state['_cluster_spec'] = collections.OrderedDict(
                sorted(ordered_state['_cluster_spec'].as_dict().items(), key=lambda t: t[0]))
        return ', '.join(
            '%s=%r' % (k, v) for (k, v) in six.iteritems(ordered_state))


class Generator(object):
    """Load generator ready to complete partial scans."""

    def __init__(self, 
                 ckpt_dir, 
                 generator_architecture=generator_architecture,
                 batch_size=1):

        tf.reset_default_graph()

        temp = set(tf.all_variables())

        # The env variable is on deprecation path, default is set to off.
        os.environ['TF_SYNC_ON_FINISH'] = '0'
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        #with tf.device("/cpu:0"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #For batch normalisation windows
        with tf.control_dependencies(update_ops):

            # Session configuration.
            sess_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                intra_op_parallelism_threads=0, #Once placement is correct, this fills up too much of the cmd window...
                gpu_options=tf.GPUOptions(force_gpu_compatible=True, allow_growth=True))

            config = RunConfig(
                session_config=sess_config, model_dir=model_dir)

            sess = tf.Session(config=sess_config)

            sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
            temp = set(tf.all_variables())

            img_ph = [tf.placeholder(tf.float32, shape=(batch_size,cropsize,cropsize, channels), name='img') 
                        for i in range(batch_size)]
            img_truth_ph = [tf.placeholder(tf.float32, shape=(batch_size,cropsize,cropsize, channels), name='img_truth') 
                        for i in range(batch_size)]
            img_mask_ph = [tf.placeholder(tf.float32, shape=(batch_size,cropsize,cropsize, channels), name='img_mask') 
                        for i in range(batch_size)]

            is_training = True

            learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate')
            discr_learning_rate_ph = tf.placeholder(tf.float32, name='discr_learning_rate')
            beta1_ph = tf.placeholder(tf.float32, shape=(), name='beta1')
            discr_beta1_ph = tf.placeholder(tf.float32, shape=(), name='discr_beta1')
            norm_decay_ph = tf.placeholder(tf.float32, shape=(), name='norm_decay')
            train_outer_ph = tf.placeholder(tf.bool, name='train_outer')
            ramp_ph = tf.placeholder(tf.float32, name='ramp')

            #########################################################################################

            exp_dict = experiment(img_ph[0], img_truth_ph[0], img_mask_ph[0], 
                                    learning_rate_ph, discr_learning_rate_ph, 
                                    beta1_ph, discr_beta1_ph, norm_decay_ph, 
                                    train_outer_ph, ramp_ph, initialize=True)

            sess.run( tf.initialize_variables( set(tf.all_variables())-temp), 
                        feed_dict={beta1_ph: np.float32(0.9), discr_beta1_ph: np.float32(0.5)} )
            train_writer = tf.summary.FileWriter( logDir, sess.graph )

            saver = tf.train.Saver(max_to_keep=1)
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

            base_dict = { learning_rate_ph: 0.0001, 
                            discr_learning_rate_ph: np.float32(0.0002),
                            beta1_ph: np.float32(0.9), 
                            discr_beta1_ph: np.float32(0.9),
                            norm_decay_ph: np.float32(1.),
                            train_outer_ph: np.bool(True),
                            ramp_ph: np.float32(1.)
                            }

            self._sess = sess
            self._base_dict = base_dict
            self._partial_scan = img_ph[0]
            self._mask = img_mask_ph[0]
            self._complete_scan = exp_dict["output_outer"]


    def infer(self, partial_scan, path, add_dims=True):
        """
        Complete partial scans with deep learning
        
        Inputs:
            partial_scan: Batch of partial scans with shape [batch_size, 512, 512, 1]. 
            Intensities should be in [-1,1]
            path: Durations that pixels are traversed by electron beam.
            add_dims: reshape partial scan to (1, 512, 512, 1)

        Returns:
            Completed partial scans in the same shape as the inputs.
        """

        #If images are 2D, add batch and feature dimensions
        if add_dims:
            orig_partial_scan_shape = partial_scan.shape
            partial_scan = np.expand_dims(np.reshape(partial_scan, (cropsize, cropsize, channels)), axis=0)

            path = np.expand_dims(np.reshape(path, (cropsize, cropsize, channels)), axis=0)

        feed_dict = self._base_dict.copy()
        feed_dict.update({ self._partial_scan: partial_scan, self._mask: path })

        completed_scans = self._sess.run(
            self._complete_scan,
            feed_dict=feed_dict)

        if add_dims:
            completed_scans = np.reshape(completed_scans, orig_partial_scan_shape)

        return completed_scans

def get_example_scan(idx=None):

    if idx == None:
        idx = random.randint(1, 5)

    #partial_scan_filepath = os.getcwd().replace("\\", "/") + f"/example_scans/partial_scan-{idx}.tif"
    #path_filepath = os.getcwd().replace("\\", "/") + f"/example_scans/mask-{idx}.tif"
    #truth_filepath = os.getcwd().replace("\\", "/") + f"/example_scans/truth-{idx}.tif"

    #partial_scan = imread(partial_scan_filepath, mode="F")
    #path = imread(path_filepath, mode="F")
    #truth = imread(truth_filepath, mode="F")

    start = r"Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-48/"
    partial_scan = imread(start+"input-2700000.tif", mode="F")
    path = imread(start+"mask-2700000.tif", mode="F")
    truth = imread(start+"truth-2700000.tif", mode="F")

    return partial_scan, path, truth

def scale0to1(img):
    """Rescale image between 0 and 1"""

    img = img.astype(np.float32)

    min = np.min(img)
    max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.5)
    else:
        img = (img - min)/(max - min)

    return img.astype(np.float32)

def disp(img):
    """Image display utility."""
    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)
    return

def img_save(img, path, dtype=np.float32):
    """Image saver utility. Does not reshape."""
    Image.fromarray(img.astype(np.float32)).save( path )
    return

def reader(seq_len, file_num):

    filename = r"\\DESKTOP-SA1EVJV\dataset\cifs\{seq_len}\{file_num}.P"
    with open(filename, "rb") as f:
        path, data = pickle.load(f)

    print(path)

if __name__ == "__main__":

    partial_scan, path, truth = get_example_scan()

    ckpt_dir = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/stem-random-walk-nin-20-48/model/"

    gen = Generator(ckpt_dir=ckpt_dir)
    disp(gen.infer(partial_scan, path))
    #img_save(gen.infer(partial_scan), "C:/dump/test.tif")