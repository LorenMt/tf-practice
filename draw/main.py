'''
Tensorflow implementation of "DRAW: A Recurrent Neural Network For Image Generation"
Original paper link: https://arxiv.org/abs/1502.04623
Author: Liu, Shikun
'''
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn, layers

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.flags.DEFINE_boolean("read_attn", True, "apply attention to model")

class DRAW(object):
    def __init__(self, input_shape,
                 batch_size,
                 latent_dim,
                 sequence_len,
                 network_architecture):
        # define model parameters
        self.input_shape  = input_shape
        self.batch_size   = batch_size
        self.latent_dim   = latent_dim
        self.sequence_len = sequence_len
        self.network_architecture = network_architecture

        # define input placeholder
        self.x = tf.placeholder(tf.float32,[None, self.input_shape])
        self.SHARE = False

        # create model and define its loss and optimizer
        self._model_create()
        self._model_loss_optimizer()

        # start tensorflow session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # initialize model weights (using dictionary style)
    def _weights_init(self, n_hidden):
        self.RNN_enc = rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)
        self.RNN_dec = rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)

        weights_all = dict()
        weights_all['W'] = {
            'write' : tf.get_variable(name='write',
                                          shape=[n_hidden, self.input_shape],
                                          initializer=layers.xavier_initializer()),
            'lat_mean'  : tf.get_variable(name='lat_mean',
                                          shape=[n_hidden, self.latent_dim],
                                          initializer=layers.xavier_initializer()),
            'lat_logstd': tf.get_variable(name='lat_logstd',
                                          shape=[n_hidden, self.latent_dim],
                                          initializer=layers.xavier_initializer())}
        weights_all['b'] = {
            'write' : tf.Variable(name='decoder',
                                      initial_value=tf.zeros(self.input_shape)),
            'lat_mean'  : tf.Variable(name='lat_mean',
                                      initial_value=tf.zeros(self.latent_dim)),
            'lat_logstd': tf.Variable(name='lat_logstd',
                                      initial_value=tf.zeros(self.latent_dim))}

        return weights_all

    # create model reader
    def _model_read(self, x, x_rec, h_dec_prev):
        return tf.concat([x, x_rec], axis=1)

    # create model writer
    def _model_write(self, x, weights, biases):
        with tf.variable_scope("model_write", reuse=self.SHARE):
            return  tf.add(tf.matmul(x, weights['write']),  biases['write'])

    # create model decoder
    def _model_decode(self, x, state):
        with tf.variable_scope("decoder", reuse=self.SHARE):
            return self.RNN_dec(x, state)

    # create model encoder
    def _model_encode(self, x, state):
        with tf.variable_scope("encoder", reuse=self.SHARE):
            return self.RNN_enc(x, state)

    # sampling z_t ~ Q(Z_t|h_enc), re-parametrization trick
    def _model_sampleQ(self, x, weights, biases):
        # sample z_mean for latent variable
        with tf.variable_scope("lat_mean", reuse=self.SHARE):
            z_mean = tf.add(tf.matmul(x, weights['lat_mean']),  biases['lat_mean'])

        # sample z_logstd for latent variable
        with tf.variable_scope("lat_logstd", reuse=self.SHARE):
            z_logstd = tf.add(tf.matmul(x, weights['lat_logstd']),  biases['lat_logstd'])

        epsilon = tf.random_normal((self.batch_size, self.latent_dim))
        return z_mean + tf.exp(z_logstd) * epsilon, z_mean, z_logstd

    # create DRAW model
    def _model_create(self):
        # load defined network structure
        network_weights = self._weights_init(**self.network_architecture)

        # define zero state for parameters
        self.c, self.z_mean, self.z_logstd = [[[0]] * self.sequence_len for _ in range(3)]
        self.enc_state = self.RNN_enc.zero_state(self.batch_size, tf.float32)
        self.dec_state = self.RNN_dec.zero_state(self.batch_size, tf.float32)
        self.h_dec_prev = tf.zeros((self.batch_size, 256))

        # define model with attention
        for t in range(self.sequence_len):
            self.c_prev     = tf.zeros((self.batch_size, self.input_shape)) if t == 0 else self.c[t - 1]
            self.x_rec = self.x - tf.sigmoid(self.c_prev)
            self.r = self._model_read(self.x, self.x_rec, self.h_dec_prev)
            self.h_enc, self.enc_state = self._model_encode(tf.concat([self.r, self.h_dec_prev], axis=1), self.enc_state)
            self.z, self.z_mean[t], self.z_logstd[t] = self._model_sampleQ(self.h_enc, network_weights['W'], network_weights['b'])
            self.h_dec, self.dec_state = self._model_decode(self.z, self.dec_state)
            self.c[t] = self.c_prev + self._model_write(self.h_dec, network_weights['W'], network_weights['b'])
            self.h_dec_prev = self.h_dec

            self.SHARE = True

        # final reconstructed image
        self.x_rec = tf.nn.sigmoid(self.c[-1])

    # define VAE loss and optimizer
    def _model_loss_optimizer(self):
        # define reconstruction loss (binary cross-entropy)
        self.rec_loss = -tf.reduce_sum(self.x * tf.log(1e-8 + self.x_rec)
                                       + (1 - self.x) * tf.log(1e-8 + 1 - self.x_rec), axis=1)

        # define kl loss KL(p(z|x)||q(z))
        self.kl_loss = [0]*self.sequence_len
        for i in range(self.sequence_len):
            self.kl_loss[i] = -0.5 * tf.reduce_sum(
                1 + 2 * self.z_logstd[i] - tf.square(self.z_mean[i]) - tf.square(tf.exp(self.z_logstd[i])), axis=1)

        self.kl_loss = tf.reduce_mean(tf.add_n(self.kl_loss))

        self.loss = tf.reduce_mean(tf.add(self.rec_loss, self.kl_loss))

        self.opt_func = tf.train.AdamOptimizer()
        self.gvs = self.opt_func.compute_gradients(self.loss)
        self.capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gvs]
        self.optimizer = self.opt_func.apply_gradients(self.capped_gvs)

    # train model on mini-batch
    def model_fit(self, x):
        opt, cost = self.sess.run([self.optimizer, self.loss], feed_dict={self.x: x})
        return cost



# load MNIST dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# load network architecture, parameters
latent_dim  = 10
batch_size  = 100
input_shape = 784
print_step  = 1
total_epoch = 5
sequence_len = 20

network_architecture = dict(n_hidden = 256)

# load VAE model
DRAW = DRAW(input_shape=input_shape,
          latent_dim=latent_dim,
          batch_size=batch_size,
          sequence_len=sequence_len,
          network_architecture=network_architecture)

# training VAE model
for epoch in range(total_epoch):
    cost     = np.zeros(3, dtype=np.float32)
    avg_cost = np.zeros(3, dtype=np.float32)
    total_batch = int(mnist.train.num_examples / batch_size)
    # iterate for all batches
    for i in range(total_batch):
        x_train, _ = mnist.train.next_batch(batch_size)
        # calculate and average kl and vae loss for each batch
        cost[0] = np.mean(DRAW.sess.run(DRAW.kl_loss, feed_dict={DRAW.x: x_train}))
        cost[1] = np.mean(DRAW.sess.run(DRAW.rec_loss, feed_dict={DRAW.x: x_train}))
        cost[2] = DRAW.model_fit(x_train)
        avg_cost += cost / mnist.train.num_examples

    if epoch % print_step == 0:
        print("Epoch: {:04d} | kl-loss: {:.4f} + rec-loss: {:.4f} = total-loss: {:.4f}"
              .format((epoch+1), avg_cost[0], avg_cost[1], avg_cost[2]))


