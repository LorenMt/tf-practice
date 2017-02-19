'''
Tensorflow implementation of "Auto-Encoding Variational Bayes"
Original paper link: https://arxiv.org/abs/1312.6114
Author: Liu, Shikun
'''
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import layers

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class VAE(object):
    def __init__(self, input_shape,
                 batch_size,
                 latent_dim,
                 network_architecture):
        # define model parameters
        self.input_shape= input_shape
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.network_architecture = network_architecture

        # define input placeholder
        self.x = tf.placeholder(tf.float32,[None, self.input_shape])

        # create model and define its loss and optimizer
        self._model_create()
        self._model_loss_optimizer()

        # start tensorflow session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # initialize model weights (using dictionary style)
    def _weights_init(self, n_hidden_1, n_hidden_2):
        weights_all = dict()
        weights_all['W'] = {
            'encoder_1' : tf.get_variable(name='encoder_1',
                                          shape=[self.input_shape, n_hidden_1],
                                          initializer=layers.xavier_initializer()),
            'encoder_2' : tf.get_variable(name='encoder_2',
                                          shape=[n_hidden_1, n_hidden_2],
                                          initializer=layers.xavier_initializer()),
            'decoder_1' : tf.get_variable(name='decoder_1',
                                          shape=[self.latent_dim, n_hidden_1],
                                          initializer=layers.xavier_initializer()),
            'decoder_2' : tf.get_variable(name='decoder_2',
                                          shape=[n_hidden_1, self.input_shape],
                                          initializer=layers.xavier_initializer()),
            'lat_mean'  : tf.get_variable(name='lat_mean',
                                          shape=[n_hidden_2, self.latent_dim],
                                          initializer=layers.xavier_initializer()),
            'lat_logstd': tf.get_variable(name='lat_logstd',
                                          shape=[n_hidden_2, self.latent_dim],
                                          initializer=layers.xavier_initializer()),
        }
        weights_all['b'] = {
            'encoder_1' : tf.Variable(name='encoder_1',
                                      initial_value=tf.zeros(n_hidden_1)),
            'encoder_2' : tf.Variable(name='encoder_2',
                                      initial_value=tf.zeros(n_hidden_2)),
            'decoder_1' : tf.Variable(name='decoder_1',
                                      initial_value=tf.zeros(n_hidden_1)),
            'decoder_2' : tf.Variable(name='decoder_2',
                                      initial_value=tf.zeros(self.input_shape)),
            'lat_mean'  : tf.Variable(name='lat_mean',
                                      initial_value=tf.zeros(self.latent_dim)),
            'lat_logstd': tf.Variable(name='lat_logstd',
                                      initial_value=tf.zeros(self.latent_dim)),
        }
        return weights_all


    # define inference model
    def _inf_model(self, x, weights, biases):
        fclayer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_1']),
                                      biases['encoder_1']))
        fclayer_2 = tf.nn.relu(tf.add(tf.matmul(fclayer_1, weights['encoder_2']),
                                      biases['encoder_2']))
        z_mean    = tf.add(tf.matmul(fclayer_2, weights['lat_mean']),
                           biases['lat_mean'])
        z_logstd  = tf.add(tf.matmul(fclayer_2, weights['lat_logstd']),
                           biases['lat_logstd'])
        return z_mean, z_logstd

    # define generative model
    def _gen_model(self, x, weights, biases):
        fclayer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_1']),
                                      biases['decoder_1']))
        fclayer_2 = tf.nn.sigmoid(tf.add(tf.matmul(fclayer_1, weights['decoder_2']),
                                         biases['decoder_2']))
        return fclayer_2

    # use learned parameters to sample latent space (z ~ N(0, I))
    def _sampling(self, z_mean, z_logstd):
        epsilon = tf.random_normal((self.batch_size, self.latent_dim))
        return z_mean + tf.exp(z_logstd) * epsilon

    # create VAE model
    def _model_create(self):
        # load defined network structure
        network_weights = self._weights_init(**self.network_architecture)

        # learn gaussian parameters from inference network
        self.z_mean, self.z_logstd = self._inf_model(self.x, network_weights['W'], network_weights['b'])

        # sampling latent space from learned parameters
        self.z = self._sampling(self.z_mean, self.z_logstd)

        # reconstruct training data from sampled latent states
        self.x_rec = self._gen_model(self.z, network_weights['W'], network_weights['b'])

    def _model_loss_optimizer(self):
        # define reconstruction loss (binary cross-entropy)
        self.rec_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_rec)
                                       + (1-self.x) * tf.log(1e-10 + 1 - self.x_rec), axis=1)

        # define kl loss KL(p(z|x)||q(z))
        self.kl_loss = -0.5 * tf.reduce_sum(1 + 2 * self.z_logstd - tf.square(self.z_mean) - tf.square(tf.exp(self.z_logstd)), axis=1)

        self.loss = tf.reduce_mean(tf.add(self.rec_loss, self.kl_loss))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)


    def model_fit(self, x):
        opt, cost = self.sess.run([self.optimizer, self.loss], feed_dict={self.x: x})
        return cost

    def model_rec(self, x):
        return self.sess.run(self.x_rec, feed_dict={self.x: x})



# load MNIST dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# load network architecture, parameters
latent_dim  = 2
batch_size  = 1000
input_shape = 784
print_step  = 1
total_epoch = 100
network_architecture = dict(n_hidden_1 = 600,
                            n_hidden_2 = 600)

# load VAE model
VAE = VAE(input_shape=784,
          latent_dim=latent_dim,
          batch_size=batch_size,
          network_architecture=network_architecture)

# training VAE model
for epoch in range(total_epoch):
    total_batch = int(mnist.train.num_examples / batch_size)
    avg_cost = 0.
    # Loop over all batches
    for i in range(total_batch):
        x_train, _ = mnist.train.next_batch(batch_size)
        # run optimization op (backprop) and cost (to get loss value)
        cost = VAE.model_fit(x_train)
        # average total cost
        avg_cost += cost / mnist.train.num_examples

    if epoch % print_step == 0:
        print("Epoch: {:04d} | total-loss: {:.4f}".format((epoch+1), avg_cost))


# reconstruction results visualization
x_sample = mnist.test.next_batch(batch_size)[0]
x_reconstruct = VAE.model_rec(x_sample)

plt.figure(figsize=(8, 12))
for i in range(5):
    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(5, 2, 2*i + 2)
    plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("Reconstruction")
    plt.colorbar()
plt.tight_layout()

# latnet space visualization
x_sample, y_sample = mnist.test.next_batch(5000)
z_mu = VAE.transform(x_sample)
plt.figure(figsize=(8, 6))
plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
plt.colorbar()
plt.grid()

