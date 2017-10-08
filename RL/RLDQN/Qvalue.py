import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


class Qvalue:
    def __init__(self, state_dim=2, n_actions=3, batch_size=64, h1_n=512, h2_n=256):
        learning_rate = 0.01
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope("DNN"):
                # TODO:Replace None with batch_size
                # TODO: solve diff between predict for 1 & predict for batch
                self.x = tf.placeholder(tf.float32, shape=(batch_size, state_dim), name="X")
                self.targets = tf.placeholder(tf.float32, shape=(batch_size, n_actions), name="targets")

                self.y = self.model(self.x, h1_n, h2_n, n_actions)
                # find better name for test_x (used for predict one state or states with batch_size len)
                self.test_x = tf.placeholder(tf.float32, shape=(None, state_dim), name="test_X")

                # self.hidden1 = fully_connected(self.x, h1_n, scope="Hdden1")
                # self.hidden2 = fully_connected(self.hidden1, h2_n, scope="hidden2")
                # self.y = fully_connected(self.hidden2, n_actions, scope="logits")

            with tf.name_scope("loss"):
                # xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
                # self.loss = tf.reduce_mean(xentropy, name="Loss")
                self.loss = tf.reduce_mean(tf.squared_difference(self.targets, self.y))

            with tf.name_scope("train"):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)

                # optimizer = tf.train.AdagradDAOptimizer(learning_rate)
                # training_op = optimizer.minimize(loss=self.loss)
            self.test_y = self.model(self.test_x, h1_n, h2_n, n_actions)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        self.session = None

    def inti_tf_session(self):
        if self.session is None:
            self.session = tf.Session(graph=self.graph)
            self.session.run(self.init)

    def model(self, data, h1_n, h2_n, n_actions):
        hidden1 = fully_connected(data, h1_n)
        hidden2 = fully_connected(hidden1, h2_n)
        return fully_connected(hidden2, n_actions)

    def predict(self, states):
        self.inti_tf_session()
        feed_dict = {self.test_x: states}

        return self.session.run(self.test_y, feed_dict=feed_dict)

    def train(self, states, targets):
        self.inti_tf_session()
        feed_dict = {self.x: states, self.targets: targets}
        return self.session.run(self.loss, feed_dict=feed_dict)
