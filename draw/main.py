'''
Tensorflow implementation of "DRAW: A Recurrent Neural Network For Image Generation"
Original paper link: https://arxiv.org/abs/1502.04623
Author: Liu, Shikun
'''
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn, layers

import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.flags.DEFINE_boolean("attention", True, "apply attention to model")

class DRAW(object):
    def __init__(self, input_shape,
                 batch_size,
                 filter_size,
                 latent_dim,
                 sequence_len,
                 network_architecture):
        # define model parameters
        self.input_shape  = input_shape
        self.filter_size  = filter_size
        self.batch_size   = batch_size
        self.latent_dim   = latent_dim
        self.sequence_len = sequence_len
        self.network_architecture = network_architecture

        # define input placeholder
        self.x = tf.placeholder(tf.float32,[None, self.input_shape])
        self.A = 28
        self.B = 28
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
        self.write_size = self.filter_size ** 2 if tf.flags.FLAGS.attention else self.input_shape

        weights_all = dict()
        weights_all['W'] = {
            'write'     : tf.get_variable(name='write',
                                          shape=[n_hidden, self.write_size],
                                          initializer=layers.xavier_initializer()),
            'attention' : tf.get_variable(name='attention',
                                          shape=[n_hidden, 5],
                                          initializer=layers.xavier_initializer()),
            'lat_mean'  : tf.get_variable(name='lat_mean',
                                          shape=[n_hidden, self.latent_dim],
                                          initializer=layers.xavier_initializer()),
            'lat_logstd': tf.get_variable(name='lat_logstd',
                                          shape=[n_hidden, self.latent_dim],
                                          initializer=layers.xavier_initializer())}
        weights_all['b'] = {
            'write'     : tf.Variable(name='write',
                                      initial_value=tf.zeros(self.write_size)),
            'attention' : tf.Variable(name='attention',
                                      initial_value=tf.zeros(5)),
            'lat_mean'  : tf.Variable(name='lat_mean',
                                      initial_value=tf.zeros(self.latent_dim)),
            'lat_logstd': tf.Variable(name='lat_logstd',
                                      initial_value=tf.zeros(self.latent_dim))}
        return weights_all

    # create attention window
    def _attention(self, mode, x, N, weights, biases):
        with tf.variable_scope(mode, reuse=self.SHARE):
            params = tf.add(tf.matmul(x, weights['attention']), biases['attention'])

        # five parameters determined by linear transformation of h_dec
        gx_, gy_, log_stdsq, log_delta, log_gamma = tf.split(params, 5, axis=1)

        # determine Gaussian filter parameters : eq(22) - eq(24)
        gx = (self.A + 1) / 2 * (gx_ + 1)
        gy = (self.B + 1) / 2 * (gy_ + 1)
        delta = (max(self.A, self.B) - 1) / (N - 1) * tf.exp(log_delta)

        # find mean location of the filter
        i, j = [tf.cast(tf.range(N), tf.float32)]*2
        mu_x = gx + (i - N/2 - 0.5) * delta
        mu_y = gy + (j - N/2 - 0.5) * delta

        # filterbank matrices
        a = tf.reshape(tf.cast(tf.range(self.A), tf.float32), [1, 1, -1])
        b = tf.reshape(tf.cast(tf.range(self.B), tf.float32), [1, 1, -1])
        mu_x = tf.reshape(mu_x, [-1, N, 1])
        mu_y = tf.reshape(mu_y, [-1, N, 1])
        log_stdsq = tf.reshape(log_stdsq, [-1, 1, 1])

        Fx = tf.exp(-tf.square(a - mu_x) / (2 * tf.exp(log_stdsq)))
        Fy = tf.exp(-tf.square(b - mu_y) / (2 * tf.exp(log_stdsq)))
        Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keep_dims=True), 10e-8)
        Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keep_dims=True), 10e-8)
        return Fx, Fy, tf.exp(log_gamma)

   # create model reader
    def _model_read(self, x, x_rec, h_dec_prev, weights, biases):
        if tf.flags.FLAGS.attention:
            Fx, Fy, gamma = self._attention('read', h_dec_prev, self.filter_size, weights, biases)

            # reshape images into A * B
            x     = tf.reshape(x, [-1, self.B, self.A])
            x_rec = tf.reshape(x_rec, [-1, self.B, self.A])

            # apply filter x = Fy*x*Fx^T, x_rec = Fy*x_rec*F_x^T
            x     = tf.einsum('aij,akj->aik', tf.einsum('aij,ajk->aik', Fy, x), Fx)
            x     = gamma * tf.reshape(x, [-1, self.filter_size ** 2])

            x_rec = tf.einsum('aij,akj->aik', tf.einsum('aij,ajk->aik', Fy, x_rec), Fx)
            x_rec = gamma * tf.reshape(x_rec, [-1, self.filter_size ** 2])
        else:
            pass
        return tf.concat([x, x_rec], axis=1)

    # create model writer
    def _model_write(self, x, weights, biases):
        if tf.flags.FLAGS.attention:
            with tf.variable_scope("write", reuse=self.SHARE):
                w_t = tf.add(tf.matmul(x, weights['write']),  biases['write'])
            w_t = tf.reshape(w_t, [-1, self.filter_size, self.filter_size])
            Fx, Fy, gamma = self._attention('write', x, self.filter_size, weights, biases)
            w_r = tf.einsum('aij,ajk->aik', tf.einsum('aji,ajk->aik', Fy, w_t), Fx)
            w_r = 1. / gamma * tf.reshape(w_r, [-1, self.input_shape])
            return w_r
        else:
            with tf.variable_scope("write", reuse=self.SHARE):
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

        # define model eq(3)-eq(8)
        for t in range(self.sequence_len):
            self.c_prev = tf.zeros((self.batch_size, self.input_shape)) if t == 0 else self.c[t - 1]

            self.x_rec = self.x - tf.sigmoid(self.c_prev)
            self.r = self._model_read(self.x, self.x_rec, self.h_dec_prev, network_weights['W'], network_weights['b'])
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

        # clip gradient avoid nan
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
latent_dim  = 20
batch_size  = 100
input_shape = 784
print_step  = 1
total_epoch = 5
sequence_len = 64
filter_size = 5
network_architecture = dict(n_hidden = 256)

# load DRAW model
DRAW = DRAW(input_shape=input_shape,
            latent_dim=latent_dim,
            batch_size=batch_size,
            sequence_len=sequence_len,
            filter_size=filter_size,
            network_architecture=network_architecture)

# training DRAW model
for epoch in range(total_epoch):
    cost     = np.zeros(3, dtype=np.float32)
    avg_cost = np.zeros(3, dtype=np.float32)
    total_batch = int(mnist.train.num_examples / batch_size)
    # iterate for all batches
    for i in range(total_batch):
        x_train, _ = mnist.train.next_batch(batch_size)
        # calculate and average kl and reconstruction loss for each batch
        cost[0] = np.mean(DRAW.sess.run(DRAW.kl_loss, feed_dict={DRAW.x: x_train}))
        cost[1] = np.mean(DRAW.sess.run(DRAW.rec_loss, feed_dict={DRAW.x: x_train}))
        cost[2] = DRAW.model_fit(x_train)
        avg_cost += cost / mnist.train.num_examples

        if epoch % print_step == 0:
            print("Epoch: {:04d} | kl-loss: {:.4f} + rec-loss: {:.4f} = total-loss: {:.4f}"
                  .format((epoch+1), cost[0], cost[1], cost[2]))



# reconstruction visualization
x_test= mnist.test.next_batch(batch_size)[0]
x_rec = DRAW.sess.run(DRAW.x_rec, feed_dict={DRAW.x:x_test})

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.axis('off')

    plt.subplot(2, 5, i +6)
    plt.imshow(x_rec[i].reshape(28, 28), cmap="gray")
    plt.axis('off')
plt.tight_layout()
plt.show()

# plot images for each t step and plot gif
def sigmoid (x):
    return 1./(1 + np.exp(-x))

c = DRAW.sess.run(DRAW.c, feed_dict={DRAW.x:x_test})
for i in range(sequence_len):
    plt.imshow(sigmoid(c[i][0].reshape(28, 28)), cmap="gray")
    plt.axis('off')
    plt.savefig("images/c_0_atten {:02d}.png".format(i), bbox_inches='tight')

    plt.imshow(sigmoid(c[i][1].reshape(28, 28)), cmap="gray")
    plt.axis('off')
    plt.savefig("images/c_1_atten {:02d}.png".format(i), bbox_inches='tight')

    plt.imshow(sigmoid(c[i][2].reshape(28, 28)), cmap="gray")
    plt.axis('off')
    plt.savefig("images/c_2_atten {:02d}.png".format(i), bbox_inches='tight')

images = []
for i in range(sequence_len):
    images.append(imageio.imread("images/c_2_atten {:02d}.png".format(i)))

imageio.mimsave('c_2_atten.gif', images)
