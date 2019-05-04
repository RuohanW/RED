'''
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
'''
import tensorflow as tf
import numpy as np

from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common import tf_util as U, fmt_row
from baselines.common.dataset_plus import Dataset, iterbatches, normalize, denormalize
from baselines import logger
def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

""" Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""
def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent

class TransitionClassifier(object):
    def __init__(self, ob_size, ac_size, hidden_size=100, log_reward=False, entcoeff=0.001, scope="adversary"):
        self.scope = scope
        self.ob_size = ob_size
        self.ac_size = ac_size
        # self.input_size = ob_size + ac_size
        self.hidden_size = hidden_size
        self.log_reward = log_reward
        self.build_ph()
        # Build grpah
        generator_logits = self.build_graph(self.generator_obs_ph, self.generator_acs_ph)
        expert_logits = self.build_graph(self.expert_obs_ph, self.expert_acs_ph)
        # Build accuracy
        generator_acc = tf.reduce_mean(tf.cast(tf.nn.sigmoid(generator_logits) < 0.5, tf.float32))
        expert_acc = tf.reduce_mean(tf.cast(tf.nn.sigmoid(expert_logits) > 0.5, tf.float32))

        weights = tf.placeholder(tf.float32, [None])
        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = weights * tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits, labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)
        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff*entropy
        # Loss + Accuracy terms
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        self.total_loss = generator_loss + expert_loss + entropy_loss
        # Build Reward for policy
        if log_reward:
            reward_op = -tf.log(1-tf.nn.sigmoid(generator_logits)+1e-8)
        else:
            reward_op = tf.nn.sigmoid(generator_logits)

        self.reward = U.function([self.generator_obs_ph, self.generator_acs_ph], reward_op)


        lr = tf.placeholder(tf.float32, None)
        self.trainer = tf.train.AdamOptimizer(learning_rate=lr)
        gvs = self.trainer.compute_gradients(self.total_loss, self.get_trainable_variables())
        self._train = U.function([self.generator_obs_ph, self.generator_acs_ph, weights, self.expert_obs_ph,
                                  self.expert_acs_ph, lr], self.losses, updates=[self.trainer.apply_gradients(gvs)])

    def build_ph(self):
        self.generator_obs_ph = tf.placeholder(tf.float32, (None, self.ob_size), name="observations_ph")
        self.generator_acs_ph = tf.placeholder(tf.float32, (None, self.ac_size), name="actions_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None, self.ob_size), name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(tf.float32, (None, self.ac_size), name="expert_actions_ph")

    def build_graph(self, obs_ph, acs_ph):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=[self.ob_size])
            obs = normalize(obs_ph, self.obs_rms)
            _input = tf.concat([obs, acs_ph], axis=1) # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=None)
        return logits

    def get_trainable_variables(self):
        return tf.trainable_variables(self.scope)

    def get_reward(self, obs, acs):
        return np.squeeze(self.reward(obs, acs))

    def build_reward_op(self, obs_ph, acs_ph):
        logits = self.build_graph(obs_ph, acs_ph)
        if self.log_reward:
            return -tf.log(1-tf.nn.sigmoid(logits)+1e-8)
        return tf.nn.sigmoid(logits)

    def set_expert_data(self, data):
        self.data = Dataset(data, deterministic=False)

    def train(self, rl_ob, rl_ac, steps=1, lr=3e-4, weights=None):
        n = rl_ob.shape[0]
        loss_buf = []
        if weights is None:
            weights = np.ones([n], dtype=np.float32)
        batch_size = rl_ob.shape[0]// steps
        for batch in iterbatches([rl_ob, rl_ac, weights], include_final_partial_batch=False, batch_size=batch_size):
            exp_ob, exp_ac = self.data.next_batch(batch_size)
            if self.obs_rms:
                self.obs_rms.update(np.concatenate([exp_ob, rl_ob], axis=0))
            loss_buf.append(self._train(*batch, exp_ob, exp_ac, lr))
        logger.info(fmt_row(13, self.loss_name))
        logger.info(fmt_row(13, np.mean(loss_buf, axis=0)))

