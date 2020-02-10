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

modelSavePeriod = 4. #Train timestep in hours
modelSavePeriod *= 3600 #Convert to s
model_dir = "//ads.warwick.ac.uk/shared/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/stem-random-walk-nin-20-72/"

shuffle_buffer_size = 5000
num_parallel_calls = 6
num_parallel_readers = 6
prefetch_buffer_size = 12
batch_size = 1
num_gpus = 1

#batch_size = 8 #Batch size to use during training
num_epochs = 1000000 #Dataset repeats indefinitely

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
use_mask = False #If true, supply mask to network as additional information
generator_input_size = cropsize
height_crop = width_crop = cropsize

discr_size = 70

weight_decay = 0.0
batch_decay_gen = 0.999
batch_decay_discr = 0.999
initial_learning_rate = 0.001
initial_discriminator_learning_rate = 0.001
num_workers = 1

increase_batch_size_by_factor = 1
effective_batch_size = increase_batch_size_by_factor*batch_size

save_result_every_n_batches = 25000

val_skip_n = 50
trainee_switch_skip_n = 1

max_num_since_training_change = 0

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

    return tf.losses.mean_squared_error(img1, img2)

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

alrc_counter = 0
def alrc(loss, num_stddev=3, decay=0.999, mu1_start=25, mu2_start=30**2):

    global alrc_counter; alrc_counter += 1

    #Varables to track first two raw moments of the loss
    mu = tf.get_variable(
        f"mu-{alrc_counter}", 
        initializer=tf.constant(mu1_start, dtype=tf.float32))
    mu2 = tf.get_variable(
        f"mu2-{alrc_counter}", 
        initializer=tf.constant(mu2_start, dtype=tf.float32))

    #Use capped loss for moment updates to limit the effect of extreme losses on the threshold
    sigma = tf.sqrt(mu2 - mu**2+1.e-8)
    loss = tf.where(loss < mu+num_stddev*sigma, 
                   loss, 
                   loss/tf.stop_gradient(loss/(mu+num_stddev*sigma)))

    #Update moments
    with tf.control_dependencies([mu.assign(decay*mu+(1-decay)*loss), mu2.assign(decay*mu2+(1-decay)*loss**2)]):
        return tf.identity(loss)

capper_counter = 0
def capper_fn(x):
    return alrc(x)
    global capper_counter; capper_counter += 1

    mu = tf.get_variable(f"mu-{capper_counter}", initializer=tf.constant(25, dtype=tf.float32))
    mu2 = tf.get_variable(f"mu2-{capper_counter}", initializer=tf.constant(30**2, dtype=tf.float32))

    def cap(x):
        sigma = tf.sqrt(mu2 - mu**2+1.e-8)
        capped_x = tf.cond(x < mu+3*sigma, lambda: x, lambda: x/tf.stop_gradient(x/(mu+3*sigma)))
        return capped_x

    x = cap(x)

    with tf.control_dependencies([mu.assign(0.999*mu+0.001*x), mu2.assign(0.999*mu2+0.001*x**2)]):
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
            x = tf.nn.leaky_relu(x, slope)
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
                    use_weight_normalization=True, use_batch_normalization=False, mean_only_norm=False,
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
        small = tf.image.resize_images(small, (discr_size, discr_size))
        medium = tf.random_crop(
                    input,
                    size=(batch_size, 2*discr_size, 2*discr_size, multiscale_channels))
        medium = tf.image.resize_images(medium, (discr_size, discr_size))
        large = tf.random_crop(
                    input,
                    size=(batch_size, 4*discr_size, 4*discr_size, multiscale_channels))
        large = tf.image.resize_images(large, (discr_size, discr_size))

        return small, medium, large

    #Generator
    feature = tf.reshape(feature, [-1, cropsize, cropsize, channels])
    feature_small = tf.image.resize_images(feature, (cropsize//2, cropsize//2))
    truth = tf.reshape(ground_truth, [-1, cropsize, cropsize, channels])
    truth_small = tf.image.resize_images(truth, (cropsize//2, cropsize//2))
    small_mask = tf.image.resize_images(mask, (cropsize//2, cropsize//2))

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

    mse_inner = 200*adjusted_mse(blur_truth_small, output_inner)
    mse_inner = capper_fn(mse_inner)
    #mse_inner = 2.*tf.cond( mse_inner < 1, lambda: mse_inner, lambda: tf.sqrt(mse_inner+1.e-8) )
    #mse_inner = tf.minimum(mse_inner, 50)

    mse_outer = 200*adjusted_mse(blur_truth, output_outer)
    mse0 = tf.reduce_mean( (blur_truth - output_outer)**2 )
    mse_outer = capper_fn(mse_outer)
    #mse_outer = 2.*tf.cond( mse_outer < 1, lambda: mse_outer, lambda: tf.sqrt(mse_outer+1.e-8) )
    #mse_outer = tf.minimum(mse_outer, 50) #Safegaurd against error spikes

    mse_outer_together = 200*adjusted_mse(blur_truth, blur_output_outer)
    mse_outer_together = capper_fn(mse_outer_together)
    #mse_outer_together = 2.*tf.cond( mse_outer < 1, lambda: mse_outer, lambda: tf.sqrt(mse_outer+1.e-8) )

    #mse_inner = 10*tf.reduce_mean(tf.abs( blur_truth_small - blur_output_inner ))
    #mse_outer = 10*tf.reduce_mean(tf.abs( blur_truth - blur_output_outer ))

    loss = mse_outer_together + wass_loss_for_gen_outer
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

    errors = tf.to_double((100*blur_truth - 100*output_outer)**2)

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
            'wass_loss_d_outer': wass_loss_for_discr_outer,
            'errors': errors,
            "mse0": mse0
            }

def flip_rotate(img):
    """Applies a random flip || rotation to the image, possibly leaving it unchanged"""

    choice = np.random.randint(0, 8)
    
    if choice == 0:
        return img
    if choice == 1:
        return np.rot90(img, 1)
    if choice == 2:
        return np.rot90(img, 2)
    if choice == 3:
        return np.rot90(img, 3)
    if choice == 4:
        return np.flip(img, 0)
    if choice == 5:
        return np.flip(img, 1)
    if choice == 6:
        return np.flip(np.rot90(img, 1), 0)
    if choice == 7:
        return np.flip(np.rot90(img, 1), 1)


def load_image(addr, resize_size=cropsize, img_type=np.float32):
    """Read an image and make sure it is of the correct type. Optionally resize it"""
    
    #addr = "Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-1/truth-1000.tif"

    try:
        img = imread(addr, mode='F')
    except:
        img = np.zeros((cropsize,cropsize))
        print("Image read failed")

    if resize_size and resize_size != cropsize:
        img = cv2.resize(img, (resize_size, resize_size), interpolation=cv2.INTER_AREA)

    return img.astype(img_type)

def scale0to1(img):
    """Rescale image between 0 and 1"""

    img = img.astype(np.float32)

    min = np.min(img)
    max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

def norm_img(img):
    
    min = np.min(img)
    max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.)
    else:
        a = 0.5*(min+max)
        b = 0.5*(max-min)

        img = (img-a) / b

    return img.astype(np.float32)

def preprocess(img):

    img[np.isnan(img)] = 0.
    img[np.isinf(img)] = 0.

    img = norm_img(img)

    return img

def gen_random_walk(channel_width, channel_height=cropsize, amplitude=1, beta1=0., shift=0., steps=10):

    walk = np.zeros((int(np.ceil(channel_width+shift)), channel_height))
    halfway = (channel_width-1)/2
    center = halfway+shift
    size = int(np.ceil(channel_width+shift))

    mom = 0.
    y = 0.
    for i in range(channel_height):

        y1 = y
        #Get new position and adjust momentum
        step_y = random.randint(0, 1)
        if step_y == 1:
            mom = beta1*mom + (1-beta1)*amplitude*(1 + np.random.normal())
            y += mom
        else:
            y = amplitude*(-1 + np.random.normal())

        if y < -halfway:
            y = -halfway
            mom = -mom
        elif y > halfway:
            y = halfway
            mom = -mom

        #Move to position in steps
        y2 = y
        scale = np.sqrt(1+(y2-y1)**2)
        for j in range(steps):
            x = (j+1)/steps
            y = (y2-y1)*x + y1

            y_idx = center+y
            if y_idx != np.ceil(y_idx):
                if int(y_idx) < size:
                    walk[int(y_idx), i] += scale*(np.ceil(y_idx) - y_idx)/steps
                if int(y_idx)+1 < size:
                    walk[int(y_idx)+1, i] += scale*(1.-(np.ceil(y_idx) - y_idx))/steps
            else:
                walk[int(y_idx), i] = scale*1

    return walk, size

#def make_mask(use_frac, amp, steps):
#    channel_size = (2+np.sqrt(4-4*4*use_frac)) / (2*use_frac)
#    num_channels = cropsize / channel_size
#    mask = np.zeros( (cropsize, cropsize) )
#    for i in range( int(num_channels) ):
#        shift = i*channel_size - np.floor(i*channel_size)
#        walk, size = gen_random_walk(channel_width=channel_size, amplitude=amp, beta1=0.5, shift=shift, steps=steps)
#        lower_idx = np.floor(i*channel_size)
#        upper_idx = int(lower_idx)+size
#        if upper_idx < cropsize:
#            mask[int(lower_idx):upper_idx, :] = walk
#        else:
#            diff = int(upper_idx)-int(cropsize)
#            mask[int(lower_idx):int(upper_idx)-diff, :] = walk[0:(size-diff), :]
#    return mask

def make_mask(use_frac):

    mask = inspiral(use_frac, cropsize)

    return mask

def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """
    #import numpy as np
    #import scipy.ndimage as nd

    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]

def gen_lq(img0):

    img = norm_img(cv2.GaussianBlur(img0,(5,5), 2.5))

    steps = 25

    use_frac = 1/49
    amp = 5.
    mask = make_mask(use_frac)
    #mask = mask.clip(0., 1.)
    #print(np.sum(mask)/(512**2))
    select = mask > 0

    #Combine with uniform noise low detection time data is less meaningful
    detection = mask*img0#mask * ( mask*img0 + 2*(1-mask)*np.random.rand(*img0.shape)*img )
    lq = -np.ones(img.shape)
    lq[select] = detection[select]
    lq = scale0to1(lq)

    lq = fill(lq, invalid=np.logical_not(mask.astype(np.bool)))

    #Changed img to img0 halfway through training
    return img0.astype(np.float32), lq.astype(np.float32), mask.astype(np.float32)

def inspiral(coverage, side, num_steps=10_000):
    """Duration spent at each location as a particle falls in a magnetic 
    field. Trajectory chosen so that the duration density is (approx.)
    evenly distributed. Trajectory is calculated stepwise.
    
    Args: 
        coverage: Average amount of time spent at a random pixel
        side: Sidelength of square image that the motion is 
        inscribed on.

    Returns:
        Amounts of time spent at each pixel on a square image as a charged
        particle inspirals.
    """
    
    #Use size that is larger than the image
    size = int(np.ceil(np.sqrt(2)*side))

    #Maximum radius of motion
    R = size/2

    #Get constant in equation of motion 
    k = 1/ (2*np.pi*coverage)

    #Maximum theta that is in the image
    theta_max = R / k

    #Equispaced steps
    theta = np.arange(0, theta_max, theta_max/num_steps)
    r = k * theta

    #Convert to cartesian, with (0,0) at the center of the image
    x = r*np.cos(theta) + R
    y = r*np.sin(theta) + R

    #Draw spiral
    z = np.empty((x.size + y.size,), dtype=x.dtype)
    z[0::2] = x
    z[1::2] = y

    z = list(z)

    img = Image.new('F', (size,size), "black")
    img_draw = ImageDraw.Draw(img)
    img_draw = img_draw.line(z)
    
    img = np.asarray(img)
    img = img[size//2-side//2:size//2+side//2+side%2, 
              size//2-side//2:size//2+side//2+side%2]

    #Blur path
    #img = cv2.GaussianBlur(img,(3,3),0)

    return img

def record_parser(record):
    """Parse files and generate lower quality images from them."""

    img = flip_rotate(preprocess(load_image(record)))
    img, lq, mask = gen_lq(img)
    if np.sum(np.isfinite(img)) != cropsize**2 or np.sum(np.isfinite(lq)) != cropsize**2:
        img = np.zeros((cropsize,cropsize))
        lq = mask*img

    return lq, img, mask

def reshaper(img1, img2, img3):
    img1 = tf.reshape(img1, [cropsize, cropsize, channels])
    img2 = tf.reshape(img2, [cropsize, cropsize, channels])
    img3 = tf.reshape(img3, [cropsize, cropsize, channels])
    return img1, img2, img3

def input_fn(dir, subset, batch_size, num_shards):
    """Create a dataset from a list of filenames and shard batches from it"""

    with tf.device('/cpu:0'):

        dataset = tf.data.Dataset.list_files(dir+subset+"/"+"*.tif")
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.map(
            lambda file: tf.py_func(record_parser, [file], [tf.float32, tf.float32, tf.float32]),
            num_parallel_calls=num_parallel_calls)
        #print(dataset.output_shapes, dataset.output_types)
        dataset = dataset.map(reshaper, num_parallel_calls=num_parallel_calls)
        #print(dataset.output_shapes, dataset.output_types)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

        iter = dataset.make_one_shot_iterator()
        img_batch = iter.get_next()

        if num_shards <= 1:
            # No GPU available or only 1 GPU.
            return [img_batch[0]], [img_batch[1]], [img_batch[2]]
        else:
            image_batch = tf.unstack(img_batch, num=batch_size, axis=1)
            feature_shards = [[] for i in range(num_shards)]
            feature_shards_truth = [[] for i in range(num_shards)]
            for i in range(batch_size):
                idx = i % num_shards
                tensors = tf.unstack(image_batch[i], num=2, axis=0)
                feature_shards[idx].append(tensors[0])
                feature_shards_truth[idx].append(tensors[1])
                feature_shards_mask[idx].append(tensors[2])
            feature_shards = [tf.parallel_stack(x) for x in feature_shards]
            feature_shards_truth = [tf.parallel_stack(x) for x in feature_shards_truth]
            feature_shards_mask = [tf.parallel_stack(x) for x in feature_shards_mask]

            return feature_shards, feature_shards_truth, feature_shards_mask

def disp(img):
    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)
    return

if disp_select:
    disp(select)

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

def sigmoid(x,shift=0,mult=1):
    return 1 / (1 + np.exp(-(x+shift)*mult))


def main(job_dir, data_dir, variable_strategy, num_gpus, log_device_placement, 
         num_intra_threads, **hparams):

    tf.reset_default_graph()

    temp = set(tf.all_variables())

    with open(log_file, 'a') as log:
        log.flush()

        # The env variable is on deprecation path, default is set to off.
        os.environ['TF_SYNC_ON_FINISH'] = '0'
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        #with tf.device("/cpu:0"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #For batch normalisation windows
        with tf.control_dependencies(update_ops):

            # Session configuration.
            sess_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,#Once placement is correct, this fills up too much of the cmd window...
                intra_op_parallelism_threads=num_intra_threads,
                gpu_options=tf.GPUOptions(force_gpu_compatible=True, allow_growth=True))

            config = RunConfig(
                session_config=sess_config, model_dir=job_dir)
            hparams=tf.contrib.training.HParams(
                is_chief=config.is_chief,
                **hparams)

            img, img_truth, img_mask = input_fn(data_dir, 'test', batch_size, num_gpus)
            img_val, img_truth_val, img_mask_val = input_fn(data_dir, 'test', batch_size, num_gpus)

            with tf.Session(config=sess_config) as sess:

                print("Session started")

                sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
                temp = set(tf.all_variables())

                ____img, ____img_truth, ____img_mask = sess.run([img, img_truth, img_mask])
                img_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img') 
                            for i in ____img]
                img_truth_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img_truth')
                                for i in ____img_truth]
                img_mask_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img_mask')
                                for i in ____img_truth]

                is_training = True

                print("Dataflow established")

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

                print("Created experiment")

                sess.run( tf.initialize_variables( set(tf.all_variables())-temp), 
                          feed_dict={beta1_ph: np.float32(0.9), discr_beta1_ph: np.float32(0.5)} )
                train_writer = tf.summary.FileWriter( logDir, sess.graph )

                #print(tf.all_variables())
                saver = tf.train.Saver(max_to_keep=1)
                #saver.restore(sess, tf.train.latest_checkpoint(model_dir+"model/"))
                saver.restore(sess, tf.train.latest_checkpoint(model_dir+"notable_ckpts/"))

                counter = 0
                val_counter = 0
                save_counter = counter
                counter_init = counter+1

                base_rate = 0.0001

                bad_buffer_size = 50
                bad_buffer_truth = []
                bad_buffer = []
                bad_buffer_mask = []
                for _ in range(bad_buffer_size):
                    lq, buffer_img, mask = sess.run([img, img_truth, img_mask])
                    bad_buffer_truth.append(buffer_img)
                    bad_buffer.append(lq)
                    bad_buffer_mask.append(mask)
                bad_buffer_prob = 0.2
                bad_buffer_beta = 0.99
                bad_buffer_thresh = 0.
                bad_buffer_tracker = bad_buffer_prob
                bad_buffer_tracker_beta = 0.99
                bad_buffer_num_uses = 1

                #Here our 'natural' statistics are MSEs
                nat_stat_mean_beta = 0.99
                nat_stat_std_dev_beta = 0.99
                nat_stat_mean = 1.5
                nat_stat2_mean = 4.

                total_iters = 1_000_000

                discr_beta1 = 0.5
                discr_learning_rate = 0.0001

                wass_iter = 1
                train_discr_per_gen = 1 #Number of discriminator training ops per generator training op

                num_steps_in_lr_decay = 8

                mses = []
                max_count = 20_000
                total_errors = None

                print("Starting training")

                while True:
                    #Train for a couple of hours
                    time0 = time.time()

                    while time.time()-time0 < modelSavePeriod:

                        if not val_counter % val_skip_n:
                            val_counter = 0
                        val_counter += 1
                        if val_counter % val_skip_n: #Only increment on nan-validation iterations

                            if not wass_iter % train_discr_per_gen:
                                counter += 1
                                wass_iter = 1
                                gen_train = True
                            else:
                                gen_train = False
                            wass_iter += 1

                        if counter < 0.25*total_iters:
                            rate = 3*base_rate
                            beta1 = 0.9
                        elif counter < 0.5*total_iters:
                            len_iters = 0.25*total_iters
                            rel_iters = counter - 0.25*total_iters
                            step = int(num_steps_in_lr_decay*rel_iters/len_iters)
                            rate = 3*base_rate * (1 - step/num_steps_in_lr_decay)

                            beta1 = 0.9 - 0.4*step/num_steps_in_lr_decay
                        #elif counter == total_iters//2:
                        #    saver.save(sess, save_path=model_dir+"model/model", global_step=counter)
                        #    quit()
                        elif counter < 0.75*total_iters:
                            rate = base_rate
                            beta1 = 0.5
                        elif counter < total_iters:
                            #Stepped linear decay
                            rel_iters = counter - 0.75*total_iters
                            step = int(num_steps_in_lr_decay*rel_iters/(0.25*total_iters))
                            rate = base_rate * ( 1. - step/num_steps_in_lr_decay )
                            beta1 = 0.5

                        if counter in [total_iters//2, total_iters]:
                            saver.save(sess, save_path=model_dir+"notable_ckpts/model", global_step=counter)
                        #if counter == total_iters:
                            quit()

                        learning_rate = np.float32(rate)

                        if counter < 0.5*total_iters:
                            norm_decay = 0.99
                        else:
                            norm_decay = 1.

                        ramp = 1.
                        train_outer = True

                        base_dict = { learning_rate_ph: learning_rate, 
                                      discr_learning_rate_ph: np.float32(discr_learning_rate),
                                      beta1_ph: np.float32(beta1), 
                                      discr_beta1_ph: np.float32(discr_beta1),
                                      norm_decay_ph: np.float32(norm_decay),
                                      train_outer_ph: np.bool(train_outer),
                                      ramp_ph: np.float32(ramp)
                                    }

                        use_buffer = False#np.random.rand() < bad_buffer_num_uses*bad_buffer_prob
                        if use_buffer:
                            idx = np.random.randint(0, bad_buffer_size)
                            _img = bad_buffer[idx]
                            _img_truth = bad_buffer_truth[idx]
                            _img_mask = bad_buffer_mask[idx]

                            print("From buffer")
                        else:
                            _img, _img_truth, _img_mask = sess.run([img, img_truth, img_mask])
                        #disp(_img_mask[0][0])
                        dict = base_dict.copy()
                        dict.update( { img_ph[0]: _img[0], img_truth_ph[0]: _img_truth[0], img_mask_ph[0]: _img_mask[0] } )

                        if counter < max_count:
                            mse = sess.run(exp_dict["mse0"], feed_dict=dict)
                            print("Iter: ", counter, mse)
                            mses.append(mse)
                            #errors = sess.run(exp_dict["errors"], feed_dict=dict)
                            #if total_errors is not None:
                            #    total_errors += errors
                            #else:
                            #    total_errors = errors
                        else:
                            mses = np.asarray(mses)
                            np.save(model_dir+"mses.npy", mses)
                            #total_errors /= max_count
                            #Image.fromarray(total_errors.reshape(cropsize, cropsize).astype(np.float32)).save( model_dir+"errors.tif" )
                            quit()

                        #if counter < max_count:
                        #    if counter == 54:
                        #        continue
                        #    print("Iter: ", counter)
                        #    final_output = sess.run(exp_dict["output_outer"], feed_dict=dict)

                        #    Image.fromarray(_img[0].reshape(cropsize, cropsize).astype(np.float32)).save( 
                        #        model_dir+f"partial_scan-{counter}.tif" )
                        #    Image.fromarray((0.5*final_output+0.5).reshape(cropsize, cropsize).astype(np.float32)).save( 
                        #        model_dir+f"output-{counter}.tif" )
                        #    Image.fromarray((0.5*_img_truth[0]+0.5).reshape(cropsize, cropsize).astype(np.float32)).save( 
                        #        model_dir+f"truth-{counter}.tif" )
                        #    Image.fromarray(_img_mask[0].reshape(cropsize, cropsize).astype(np.float32)).save( 
                        #        model_dir+f"mask-{counter}.tif" )
                        #else:
                        #    quit()

                        #if counter < 0.5*total_iters:
                        #    train_ops = exp_dict['start_train_ops']
                        #else:
                        #    train_ops = exp_dict['end_train_ops'] if gen_train else []
                        #    train_ops += exp_dict['train_ops_discr']

                        #other_ops = [exp_dict['mse_inner'], exp_dict['mse_outer'], exp_dict['wass_loss_outer'], exp_dict['wass_loss_d_outer']]
                        #output_ops = [exp_dict['output_outer']]
                        #output_size = cropsize

                        ##Save outputs occasionally
                        #if 0 <= counter <= 1 or not counter % save_result_every_n_batches or (0 <= counter < 10000 and not counter % 1000) or counter == counter_init:

                        #    #Don't train on validation examples
                        #    if not val_counter % val_skip_n:
                        #        results = sess.run( other_ops + output_ops, feed_dict=dict )
                        #    else:
                        #        results = sess.run( other_ops + output_ops + train_ops, feed_dict=dict )

                        #    mse_in = results[0]
                        #    mse = results[1]
                        #    wass_loss = results[2]
                        #    wass_d_loss = results[3]

                        #    output = results[len(other_ops)]

                        #    try:
                        #        save_input_loc = model_dir+"input-"+str(counter)+".tif"
                        #        save_truth_loc = model_dir+"truth-"+str(counter)+".tif"
                        #        save_output_loc = model_dir+"output-"+str(counter)+".tif"
                        #        save_mask_loc = model_dir+"mask-"+str(counter)+".tif"
                        #        Image.fromarray((_img[0]).reshape(cropsize, cropsize).astype(np.float32)).save( save_input_loc )
                        #        Image.fromarray((0.5*_img_truth[0]+0.5).reshape(cropsize, cropsize).astype(np.float32)).save( save_truth_loc )
                        #        Image.fromarray((0.5*output+0.5).reshape(output_size, output_size).astype(np.float32)).save( save_output_loc )
                        #        Image.fromarray((_img_mask[0]).reshape(cropsize, cropsize).astype(np.float32)).save( save_mask_loc )
                        #    except:
                        #        print("Image save failed")
                        #else:
                        #    #Don't train on validation examples
                        #    if not val_counter % val_skip_n:
                        #        results = sess.run( other_ops, feed_dict=dict )
                        #    else:
                        #        results = sess.run( other_ops + train_ops, feed_dict=dict )

                        #    mse_in = results[0]
                        #    mse = results[1]
                        #    wass_loss = results[2]
                        #    wass_d_loss = results[3]

                        #nat_stat_mean = (nat_stat_mean_beta*nat_stat_mean + 
                        #                 (1.-nat_stat_mean_beta)*mse)
                        #nat_stat2_mean = (nat_stat_std_dev_beta*nat_stat2_mean + 
                        #                  (1.-nat_stat_std_dev_beta)*mse**2)

                        #nat_stat_std_dev = np.sqrt(nat_stat2_mean - nat_stat_mean**2)

                        ##Decide whether or not to add to buffer using natural statistics
                        #if not use_buffer and mse > bad_buffer_thresh:
                        #    idx = np.random.randint(0, bad_buffer_size)
                        #    bad_buffer[idx] = _img
                        #    bad_buffer_truth[idx] = _img_truth
                        #    bad_buffer_mask[idx] = _img_mask
                            
                        #    bad_buffer_tracker = ( bad_buffer_tracker_beta*bad_buffer_tracker + 
                        #                           (1.-bad_buffer_tracker_beta) )
                        #    print("To buffer")#, bad_buffer_thresh, bad_buffer_prob, bad_buffer_tracker)
                        #else:
                        #    bad_buffer_tracker = bad_buffer_tracker_beta*bad_buffer_tracker

                        #if bad_buffer_tracker < bad_buffer_prob:
                        #    step = nat_stat_mean-5*nat_stat_std_dev
                        #    bad_buffer_thresh = bad_buffer_beta*bad_buffer_thresh + (1.-bad_buffer_beta)*step

                        #if bad_buffer_tracker >= bad_buffer_prob:
                        #    step = nat_stat_mean+5*nat_stat_std_dev
                        #    bad_buffer_thresh = bad_buffer_beta*bad_buffer_thresh + (1.-bad_buffer_beta)*step

                        #message = "NiN-44, Iter: {}, MSE_in: {}, MSE: {}, Wass G: {}, Wass D: {}, Val: {}".format(
                        #    counter, 3.5/2*mse_in,  3.5/2*mse, wass_loss, wass_d_loss, 
                        #    1 if not val_counter % val_skip_n else 0)
                        #print(message)
                        #try:
                        #    log.write(message)
                        #except:
                        #    print("Write to log failed")

                    #Save the model
                    #saver.save(sess, save_path=model_dir+"model/model", global_step=counter)
                    save_counter = counter
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        default=data_dir,
        help='The directory where the CIFAR-10 input data is stored.')
    parser.add_argument(
        '--job-dir',
        type=str,
        default=model_dir,
        help='The directory where the model will be stored.')
    parser.add_argument(
        '--variable-strategy',
        choices=['CPU', 'GPU'],
        type=str,
        default='GPU',
        help='Where to locate variable operations')
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=num_gpus,
        help='The number of gpus used. Uses only CPU if set to 0.')
    parser.add_argument(
        '--log-device-placement',
        action='store_true',
        default=True,
        help='Whether to log device placement.')
    parser.add_argument(
        '--num-intra-threads',
        type=int,
        default=0,
        help="""\
        Number of threads to use for intra-op parallelism. When training on CPU
        set to 0 to have the system pick the appropriate number or alternatively
        set it to the number of physical CPU cores.\
        """)
    parser.add_argument(
        '--train-steps',
        type=int,
        default=80000,
        help='The number of steps to use for training.')
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=batch_size,
        help='Batch size for training.')
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=batch_size,
        help='Batch size for validation.')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Momentum for MomentumOptimizer.')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help="""\
        This is the inital learning rate value. The learning rate will decrease
        during training. For more details check the model_fn implementation in
        this file.\
        """)
    parser.add_argument(
        '--sync',
        action='store_true',
        default=False,
        help="""\
        If present when running in a distributed environment will run on sync mode.\
        """)
    parser.add_argument(
        '--num-inter-threads',
        type=int,
        default=0,
        help="""\
        Number of threads to use for inter-op parallelism. If set to 0, the
        system will pick an appropriate number.\
        """)
    parser.add_argument(
        '--data-format',
        type=str,
        default="NHWC",
        help="""\
        If not set, the data format best for the training device is used. 
        Allowed values: channels_first (NCHW) channels_last (NHWC).\
        """)
    parser.add_argument(
        '--batch-norm-decay',
        type=float,
        default=0.997,
        help='Decay for batch norm.')
    parser.add_argument(
        '--batch-norm-epsilon',
        type=float,
        default=1e-5,
        help='Epsilon for batch norm.')
    args = parser.parse_args()

    if args.num_gpus > 0:
        assert tf.test.is_gpu_available(), "Requested GPUs but none found."
    if args.num_gpus < 0:
        raise ValueError(
            'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')
    if args.num_gpus == 0 and args.variable_strategy == 'GPU':
        raise ValueError('num-gpus=0, CPU must be used as parameter server. Set'
                         '--variable-strategy=CPU.')
    if args.num_gpus != 0 and args.train_batch_size % args.num_gpus != 0:
        raise ValueError('--train-batch-size must be multiple of --num-gpus.')
    if args.num_gpus != 0 and args.eval_batch_size % args.num_gpus != 0:
        raise ValueError('--eval-batch-size must be multiple of --num-gpus.')

    main(**vars(args))
