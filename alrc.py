"""
Implementation of adaptive learning rate clipping (ALRC).

ALRC is applied to each loss in a batch individually. It can 
be applied to losses with arbitrary shapes.

Implementation is `clippled_loss = alrc(loss)`. Optionally, alrc hyperparameters 
can be adjusted. Notably, performance may be improved at the start of training
if the first raw moments of the momentum are on the scale of the losses.
"""

import tensorflow as tf

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