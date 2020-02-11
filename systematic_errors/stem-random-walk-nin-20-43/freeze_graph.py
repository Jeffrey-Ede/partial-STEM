import tensorflow as tf
from tensorflow.python.tools import freeze_graph

from tensorflow.python.tools import optimize_for_inference_lib

import numpy as np
from scipy.misc import imread

import cv2

model_dir = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/stem-random-walk-nin-20-48/"

def freeze():
    freeze_graph.freeze_graph(model_dir+'frozen/partial-STEM.pbtxt', "", False, 
                              model_dir+'frozen/partial-STEM.ckpt', "Network_1/Outer/conv38/BiasAdd", #Biases added to last gen layer in this model (does not help)
                              "save/restore_all", "save/Const:0",
                              model_dir+'frozen/frozen_partial-STEM.pb', True, ""  
                             )

def optimize():

    inputGraph = tf.GraphDef()
    with tf.gfile.Open(model_dir+'frozen/frozen_partial-STEM.pb', "rb") as f:
        data2read = f.read()
        inputGraph.ParseFromString(data2read)
 
    outputGraph = optimize_for_inference_lib.optimize_for_inference(
                  inputGraph,
                  [], # an array of the input node(s)
                  ["Network_1/Outer/conv38/BiasAdd"], # an array of output nodes
                  tf.int32.as_datatype_enum)

    # Save the optimized graph'test.pb'
    f = tf.gfile.FastGFile(model_dir+'frozen/optimized_partial-STEM.pb', "w")
    f.write(outputGraph.SerializeToString()) 


def infer():
    sess = tf.Session()
    graph = tf.get_default_graph()
    with graph.as_default():
        with sess.as_default():
            #restoring the model
            saver = tf.train.import_meta_graph(model_dir+'frozen/partial-STEM.ckpt.meta')
            saver.restore(sess,tf.train.latest_checkpoint(model_dir+'frozen/'))
            #initializing all variables
            sess.run(tf.global_variables_initializer())
       
            #using the model for prediction
            partial_scan_ph = graph.get_tensor_by_name("Reshape:0")
            mask_ph = graph.get_tensor_by_name("Identity:0")
            norm_decay_ph = graph.get_tensor_by_name("mul:0")

            output = graph.get_tensor_by_name("Network_1/Outer/conv38/BiasAdd:0")

            start = r"Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-48/"
            partial_scan = imread(start+"input-2700000.tif", mode="F")
            path = imread(start+"mask-2700000.tif", mode="F")
            truth = imread(start+"truth-2700000.tif", mode="F")

            partial_scan = partial_scan[np.newaxis,...,np.newaxis]
            path = path[np.newaxis,...,np.newaxis]
            print(np.min(partial_scan), np.max(partial_scan))
            disp(path[0])
            feed_dict={partial_scan_ph: partial_scan, mask_ph: path, norm_decay_ph: np.float32(1.0)}
        
            output = sess.run(output, feed_dict)

            disp(output[0])


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


if __name__ == "__main__":
    
    #freeze()
    #optimize()
    infer()