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
                                          initializer=layers.xavier_initializer())}
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
                                      initial_value=tf.zeros(self.latent_dim))}
        return weights_all


    # define inference model (q(z|x))
    def _inf_model(self, weights, biases):
        fclayer_1 = tf.nn.relu(tf.add(tf.matmul(self.x, weights['encoder_1']),
                                      biases['encoder_1']))
        fclayer_2 = tf.nn.relu(tf.add(tf.matmul(fclayer_1, weights['encoder_2']),
                                      biases['encoder_2']))
        z_mean    = tf.add(tf.matmul(fclayer_2, weights['lat_mean']),
                           biases['lat_mean'])
        z_logstd  = tf.add(tf.matmul(fclayer_2, weights['lat_logstd']),
                           biases['lat_logstd'])
        return z_mean, z_logstd

    # define generative model (p(x|z))
    def _gen_model(self, weights, biases):
        fclayer_1 = tf.nn.relu(tf.add(tf.matmul(self.z, weights['decoder_1']),
                                      biases['decoder_1']))
        fclayer_2 = tf.nn.sigmoid(tf.add(tf.matmul(fclayer_1, weights['decoder_2']),
                                         biases['decoder_2']))
        return fclayer_2

    # use re-parametrization trick (p(z) ~ N(0, I))
    def _sampling(self, z_mean, z_logstd):
        epsilon = tf.random_normal((self.batch_size, self.latent_dim))
        return z_mean + tf.exp(z_logstd) * epsilon

    # create VAE model
    def _model_create(self):
        # load defined network structure
        network_weights = self._weights_init(**self.network_architecture)

        # learn gaussian parameters from inference network
        self.z_mean, self.z_logstd = self._inf_model(network_weights['W'], network_weights['b'])

        # sampling latent space from learned parameters
        self.z = self._sampling(self.z_mean, self.z_logstd)

        # reconstruct training data from sampled latent states
        self.x_rec = self._gen_model(network_weights['W'], network_weights['b'])

    # define VAE loss and optimizer
    def _model_loss_optimizer(self):
        # define reconstruction loss (binary cross-entropy)
        self.rec_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_rec)
                                       + (1-self.x) * tf.log(1e-10 + 1 - self.x_rec), axis=1)

        # define kl loss KL(p(z|x)||q(z))
        self.kl_loss = -0.5 * tf.reduce_sum(1 + 2 * self.z_logstd - tf.square(self.z_mean) - tf.square(tf.exp(self.z_logstd)), axis=1)

        # total loss = kl loss + rec loss
        self.loss = tf.reduce_mean(tf.add(self.rec_loss, self.kl_loss))

        # gradient clipping avoid nan
        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -1, 1)
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
latent_dim  = 2
batch_size  = 100
input_shape = 784
print_step  = 1
total_epoch = 500
network_architecture = dict(n_hidden_1 = 600,
                            n_hidden_2 = 600)

# load VAE model
VAE = VAE(input_shape=input_shape,
          latent_dim=latent_dim,
          batch_size=batch_size,
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
        cost[0] = np.mean(VAE.sess.run(VAE.kl_loss, feed_dict={VAE.x: x_train}))
        cost[1] = np.mean(VAE.sess.run(VAE.rec_loss, feed_dict={VAE.x: x_train}))
        cost[2] = VAE.model_fit(x_train)
        avg_cost += cost / mnist.train.num_examples

    if epoch % print_step == 0:
        print("Epoch: {:04d} | kl-loss: {:.4f} + rec-loss: {:.4f} = total-loss: {:.4f}"
              .format((epoch+1), avg_cost[0], avg_cost[1], avg_cost[2]))


# reconstruction results visualization
x_test= mnist.test.next_batch(batch_size)[0]
x_rec = VAE.sess.run(VAE.x_rec, feed_dict={VAE.x:x_test})

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

# latent space visualization for latent_dim=2
plt.figure(figsize=(8, 6))
for i in range(20):
    x_test, y_test= mnist.test.next_batch(batch_size)
    z = VAE.sess.run(VAE.z, feed_dict={VAE.x:x_test})
    plt.scatter(z[:, 0], z[:, 1],
                c=np.argmax(y_test, 1),
                cmap='Spectral',
                edgecolors='black')
plt.colorbar()
plt.show()


# sampling latent space with 15 * 15 examples
nx = ny = 15
x_values = np.linspace(-2, 2, nx)
y_values = np.linspace(-2, 2, ny)

canvas = np.empty((28*ny, 28*nx))
for i, yi in enumerate(x_values):
    for j, xi in enumerate(y_values):
        z = np.array([[xi, yi]]*batch_size)
        x_rec = VAE.sess.run(VAE.x_rec, feed_dict={VAE.z:z})
        canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_rec[0].reshape(28, 28)

plt.figure(figsize=(6, 6))
Xi, Yi = np.meshgrid(x_values, y_values)
plt.imshow(canvas, origin="upper", cmap="gray")
plt.axis('off')
plt.tight_layout()
plt.show()


