import tensorflow as tf
from baselines.common import tf_util as U
from baselines.common.dataset import iterbatches
from baselines import logger

class RND_Critic(object):
    def __init__(self, ob_size, ac_size, rnd_hid_size=128, rnd_hid_layer=4, hid_size=128, hid_layer=1,
                 out_size=128, scale=250000.0, offset=0., reward_scale=1.0, scope="rnd"):
        self.scope = scope
        self.scale = scale
        self.offset = offset
        self.out_size = out_size
        self.rnd_hid_size = rnd_hid_size
        self.rnd_hid_layer = rnd_hid_layer
        self.hid_size = hid_size
        self.hid_layer = hid_layer
        self.reward_scale = reward_scale
        print("RND Critic")

        ob = tf.placeholder(tf.float32, [None, ob_size])
        ac = tf.placeholder(tf.float32, [None, ac_size])
        lr = tf.placeholder(tf.float32, None)


        feat = self.build_graph(ob, ac, self.scope, hid_layer, hid_size, out_size)
        rnd_feat = self.build_graph(ob, ac, self.scope+"_rnd", rnd_hid_layer, rnd_hid_size, out_size)

        feat_loss = tf.reduce_mean(tf.square(feat-rnd_feat))
        self.reward = reward_scale*tf.exp(offset- tf.reduce_mean(tf.square(feat - rnd_feat), axis=-1) * self.scale)

        rnd_loss = tf.reduce_mean(tf.square(feat - rnd_feat), axis=-1) * self.scale
        # self.reward = reward_scale * tf.exp(offset - rnd_loss)
        # self.reward = reward_scale * (tf.math.softplus(rnd_loss) - rnd_loss)
        self.reward_func = U.function([ob, ac], self.reward)
        self.raw_reward = U.function([ob, ac], rnd_loss)

        self.trainer = tf.train.AdamOptimizer(learning_rate=lr)

        gvs = self.trainer.compute_gradients(feat_loss, self.get_trainable_variables())

        self._train = U.function([ob, ac, lr], [], updates=[self.trainer.apply_gradients(gvs)])

    def build_graph(self, ob, ac, scope, hid_layer, hid_size, size):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            layer = tf.concat([ob, ac], axis=1)
            for _ in range(hid_layer):
                layer = tf.layers.dense(layer, hid_size, activation=tf.nn.leaky_relu)
            layer = tf.layers.dense(layer, size, activation=None)
        return layer

    def build_reward_op(self, ob, ac):
        feat = self.build_graph(ob, ac, self.scope, self.hid_layer, self.hid_size, self.out_size)
        rnd_feat = self.build_graph(ob, ac, self.scope + "_rnd", self.rnd_hid_layer, self.rnd_hid_size
                                    , self.out_size)

        reward = self.reward_scale* tf.exp(self.offset- tf.reduce_mean(tf.square(feat - rnd_feat), axis=-1) * self.scale)
        return reward

    def get_trainable_variables(self):
        return tf.trainable_variables(self.scope)


    def get_reward(self, ob, ac):
        return self.reward_func(ob, ac)
    
    
    def get_raw_reward(self, ob, ac):
        return self.raw_reward(ob, ac)

    def train(self, ob, ac, batch_size=32, lr=0.001, iter=200):
        logger.info("Training RND Critic")
        for _ in range(iter):
            for data in iterbatches([ob, ac], batch_size=batch_size, include_final_partial_batch=True):
                self._train(*data, lr)


class Enc_Critic(object):
    def __init__(self, ob_size, ac_size, hid_size=128, hid_layer=1, scale=250000.0, offset=0., reward_scale=1.0,
                 reg_scale=0.0001, scope="enc"):
        self.scope = scope
        self.scale = scale
        self.offset = offset
        self.out_size = ob_size+ac_size
        self.hid_size = hid_size
        self.hid_layer = hid_layer
        self.reward_scale = reward_scale
        print("Enc Critic")

        ob = tf.placeholder(tf.float32, [None, ob_size])
        ac = tf.placeholder(tf.float32, [None, ac_size])
        lr = tf.placeholder(tf.float32, None)

        target = tf.concat([ob, ac], axis=1)
        feat = self.build_graph(ob, ac, self.scope, hid_layer, hid_size, self.out_size)


        feat_loss = tf.reduce_mean(tf.square(feat - target))
        self.reward = reward_scale * tf.exp(offset - tf.reduce_mean(tf.square(feat - target), axis=-1) * self.scale)

        raw_loss = tf.reduce_mean(tf.square(feat - target), axis=-1) * self.scale
        self.reward_func = U.function([ob, ac], self.reward)
        self.raw_reward = U.function([ob, ac], raw_loss)

        self.trainer = tf.train.AdamOptimizer(learning_rate=lr)
        if reg_scale>0:
            feat_loss +=tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(reg_scale),
                                                   weights_list=self.get_trainable_variables())

        gvs = self.trainer.compute_gradients(feat_loss, self.get_trainable_variables())

        self._train = U.function([ob, ac, lr], [], updates=[self.trainer.apply_gradients(gvs)])

    def build_graph(self, ob, ac, scope, hid_layer, hid_size, size):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            layer = tf.concat([ob, ac], axis=1)
            for _ in range(hid_layer):
                layer = tf.layers.dense(layer, hid_size, activation=tf.nn.leaky_relu)
            layer = tf.layers.dense(layer, size, activation=None)
        return layer

    def build_reward_op(self, ob, ac):
        feat = self.build_graph(ob, ac, self.scope, self.hid_layer, self.hid_size, self.out_size)
        target = tf.concat([ob, ac], axis=1)
        reward = self.reward_scale * tf.exp(
            self.offset - tf.reduce_mean(tf.square(feat - target), axis=-1) * self.scale)
        return reward

    def get_trainable_variables(self):
        return tf.trainable_variables(self.scope)

    def get_reward(self, ob, ac):
        return self.reward_func(ob, ac)

    def get_raw_reward(self, ob, ac):
        return self.raw_reward(ob, ac)

    def train(self, ob, ac, batch_size=32, lr=0.001, iter=200):
        logger.info("Training RND Critic")
        for _ in range(iter):
            for data in iterbatches([ob, ac], batch_size=batch_size, include_final_partial_batch=True):
                self._train(*data, lr)

