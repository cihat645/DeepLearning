import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # reduces logging clutter in output

def cell_forward_prop(xt, prev_a, param):
    """
    Runs forward propagation for element t of input sequence for a cell of the RNN
    :param xt: element t of input sequence
    :param a_prev: Activations from previous timestep.
    :param param: Dictionary of necessary parameters for computing y_hat
    :return: (next_a, yt_hat, cache)
            Where:   next_a - activations to be passed to next time step
                     yt_hat - predicted y values for this time step
                     cache - tuple of values we need for backprop (next_a, prev_a, xt, param)
    """
    # Retrieve Parameters
    Waa = param['Waa']
    Wax = param['Wax']
    Wya = param['Wya']
    ba = param['ba']
    by = param['by']

    # Create necessary vars

    # with tf.Session() as sess:
    #     next_a = tf.tanh(tf.matmul(Waa, prev_a) + tf.matmul(Wax, xt) + ba)  # computing activations for this time step, which will be passed to next unit
    #     yt_hat = tf.nn.softmax(tf.matmul(Wya, next_a) + by)  # Predicting y_hat
    #     cache = (next_a, prev_a, xt, param)

    next_a = tf.tanh(tf.matmul(Waa, prev_a) + tf.matmul(Wax, xt) + ba)  # computing activations for this time step, which will be passed to next unit
    yt_hat = tf.nn.softmax(tf.matmul(Wya, next_a) + by)  # Predicting y_hat
    cache = (next_a, prev_a, xt, param)

    return next_a, yt_hat, cache

# == SETTING UP TEST====
# np.random.seed(1)
# xt_npy = np.random.randn(3,10)
# a_prevnp = np.random.randn(5,10)
# Waanp = np.random.randn(5,5)
# Waxnp = np.random.randn(5,3)
# Wyanp = np.random.randn(2,5)
# banp = np.random.randn(5,1)
# bynp = np.random.randn(2,1)
#
#
# # xt_tf = tf.placeholder('xt', shape = (3,10), initializer = , trainable = True)  # how we initialize when NOT TESTING
# #prev_a = tf.get_variable('prev_a', shape = ....) # TODO: in actual implementation, use a placeholder for the data features and labels. Variables are used for the parameters
#
# tf.set_random_seed(1)
#
# xt = tf.Variable(xt_npy)
# prev_a = tf.Variable(a_prevnp)
# Waa = tf.Variable(Waanp)
# Wax = tf.Variable(Waxnp)
# Wya = tf.Variable(Wyanp)
# ba = tf.Variable(banp)
# by = tf.Variable(bynp)
#
#
# # TODO ADD THESE VARIABLES TO COLLECTION OF TRAINABLE VARS
# parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
# #
# # a_next, yt_hat, cache = rnn_forward_prop(xt, a_prev, parameters)
# # print("a_next[4] = ", a_next[4])
# # print("a_next.shape = ", a_next.shape)
# # print("yt_pred[1] =", yt_pred[1])
# # print("yt_pred.shape = ", yt_pred.shape)
# # print("\n ------ END NUMPY ------\n\n\n")
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     next_a, yt_hat, cache = cell_forward_prop(xt, prev_a, parameters)
#     print("next_a[4] (TF) = ", sess.run(next_a[4]))
#     print("next_a.shape (TF) = ", next_a.shape)
#     print("Next a type (TF) = ", type(next_a))
#     print("yt_hat[1] (TF) = ", sess.run(yt_hat[1]))
#     print("yt_hat.shape (TF) = ", yt_hat.shape)
#     # NOTE: The softmax function in the assignment doesn't normalize the values s.t. their sum = 1
#

#------------------------------END TEST 1 ------------------------------

def rnn_forward_prop(x, initial_a, param):
    """

    Note:   T_x = len(input_seq)
            n_x = # of input features for each element of the sequence
            m = # of training examples
            n_a = # of hidden units

    :param x: input sequences for dataset. Shape: (n_x, m, T_x)
                            - e.g. for text processing & using vocabulary dictionary of 5,000 words, n_x = 5000 as input at each timestep is a one-hot vector w/ len = 5000

    :param initial_a:  initial activation values to be fed into RNN. Shape: (n_a, m)

    :param param:       dictionary of weights and biases for this network

    :return:    a - The activations of the hidden units for each time step. Shape: (n_a, m, T_x)
    """
    cache_hist = [] # history of caches
    n_x, m, T_x = x.shape           # get dimensions
    n_y, n_a = param['Wya'].shape   # get dimensions

    next_a = initial_a
    init = tf.global_variables_initializer()
    activation_list = [] ; y_hat_list = []  # list of tensor objects which will be stacked together at end

    with tf.Session() as sess:
        sess.run(init)                  # initialize tf vars
        for t in range(T_x):            # for each time step in the sequence,
            next_a, y_hat, cache = cell_forward_prop(x[:, :, t], next_a, param)      # run forward prop in cell
            activation_list.append(next_a)      # store activation values
            y_hat_list.append(y_hat)              # store predicted target values
            cache_hist.append(cache)

        y_hat = tf.stack(y_hat_list, axis = 2)
        a = tf.stack(activation_list, axis = 2)
        assert y_hat.shape == (n_y, m, T_x)
        assert a.shape == (n_a, m, T_x)
        cache_hist = (cache_hist, x)

    return a, y_hat, cache_hist


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)



# ---------------------- TEST 2 -----------------------
np.random.seed(1)
x = np.random.randn(3, 10, 4)
a0 = np.random.randn(5, 10)
Waa = np.random.randn(5, 5)
Wax = np.random.randn(5, 3)
Wya = np.random.randn(2, 5)
ba = np.random.randn(5, 1)
by = np.random.randn(2,1)
# parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}


tf.set_random_seed(1)
# x_tf = tf.placeholder(dtype= 'float', shape= x.shape, name= "x")a
x_tf = tf.Variable(x)
a0_tf = tf.Variable(a0)
Waa_tf = tf.Variable(Waa)
Wax_tf = tf.Variable(Wax)
Wya_tf = tf.Variable(Wya)
ba_tf = tf.Variable(ba)
by_tf = tf.Variable(by)

parameters = {"Waa": Waa_tf, "Wax": Wax_tf, "Wya": Wya_tf, "ba": ba_tf, "by": by_tf}

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    a, y_pred, caches = rnn_forward_prop(x_tf, a0_tf, parameters)
    print("a[4][1] = ", a[4][1])
    print("a.shape = ", a.shape)
    print("y_pred[1][3] =", y_pred[1][3])
    print("y_pred.shape = ", y_pred.shape)
    print("caches[1][1][3] =", caches[1][1][3])
    print("len(caches) = ", len(caches))
# ---------------------- END TEST 2 -----------------------