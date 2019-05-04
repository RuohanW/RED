import numpy as np
import tensorflow as tf
from baselines.common import tf_util as U



def shift_up(x):
    return x-np.min(x)

class MMD_Critic(object):
    def __init__(self, ob_size, ac_size, expert_data, reward_scale=1):
        self.expert_data = expert_data
        self.b1 = np.median(self._l2_distance(expert_data))

        self.expert_tensor = tf.convert_to_tensor(expert_data, tf.float32)
        self.ob = tf.placeholder(tf.float32, shape=[None, expert_data.shape[1]])
        self.b2 = None
        self.b2_tf = tf.placeholder(tf.float32)
        self.reward_scale = reward_scale

        ob_tf = tf.placeholder(tf.float32, shape=[None, ob_size])
        ac_tf = tf.placeholder(tf.float32, shape=[None, ac_size])
        in_tf = tf.concat([ob_tf, ac_tf], axis=1)

        reward = self.build_reward_op(in_tf, self.b2_tf)
        self.reward_func = U.function([ob_tf, ac_tf, self.b2_tf], reward)


    def set_b2(self, obs, acs):
        if self.b2 is None:
            rl_data = np.concatenate([obs, acs], axis=1)
            self.b2 = np.median(self._l2_distance(rl_data, base=self.expert_data))


    def _l2_distance(self, data, base=None):
        if base is None:
            base = data
        n = data.shape[0]
        a = np.expand_dims(data, axis=1) #nx1xk
        b = np.expand_dims(base, axis=0) #1xmxk
        l2_dist = np.sum(np.square(a-b), axis=-1)
        return l2_dist

    def _l2_distance_tf(self, data, base=None):
        if base is None:
            base = data
        n = data.shape[0]
        a = tf.expand_dims(data, axis=1) #nx1xk
        b = tf.expand_dims(base, axis=0) #1xmxk
        l2_dist = tf.reduce_sum(tf.square(a-b), axis=-1)
        return l2_dist

    def get_reward(self, obs, acs, verbose=False):
        if obs.ndim == 1:
            return 0 #a shortcut to single reward as shift_up would make it zero anyway
            # obs = np.expand_dims(obs, axis=0)
            # acs = np.expand_dims(acs, axis=0)
        if self.b2 is not None:
            reward = self.reward_func(obs, acs, self.b2)
            return np.squeeze(shift_up(reward))
        else:
            return 0

    def build_reward_op(self, ob, mmd_b2):
        expert_l2 = self._l2_distance_tf(ob, self.expert_tensor)
        rl_l2 = self._l2_distance_tf(ob)
        expert_exp = tf.exp(-expert_l2 / self.b1) + tf.exp(-expert_l2 / mmd_b2)
        rl_exp = tf.exp(-rl_l2 / mmd_b2) + tf.exp(-rl_l2 / self.b1)
        reward = tf.reduce_mean(expert_exp, axis=-1) - tf.reduce_mean(rl_exp, axis=-1)
        return reward*self.reward_scale