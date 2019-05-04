'''
The code is used to train BC imitator, or pretrained GAIL imitator
'''

import argparse
import tempfile
import os.path as osp
import gym
import logging

import tensorflow as tf

from baselines.rnd_gail import mlp_policy
from baselines.common import set_global_seeds, tf_util as U, dataset
from baselines.common.misc_util import boolean_flag
from baselines.common.mpi_adam import MpiAdam
from baselines.common.dataset_plus import iterbatches




def learn(env, policy_func, dataset, task_name, optim_batch_size=128, max_iters=1e4,
          adam_epsilon=1e-5, optim_stepsize=3e-4, ckpt_dir=None):
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
    # placeholder
    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])
    stochastic = U.get_placeholder_cached(name="stochastic")
    loss = tf.reduce_mean(tf.square(ac-pi.ac))
    var_list = pi.get_trainable_variables()
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    lossandgrad = U.function([ob, ac, stochastic], [loss]+[U.flatgrad(loss, var_list)])


    U.initialize()
    adam.sync()

    if hasattr(pi, "ob_rms"): pi.ob_rms.update(dataset[0])
    for _ in range(int(max_iters)):
        for batch in iterbatches(dataset, batch_size=optim_batch_size):
            train_loss, g = lossandgrad(*batch, True)
            adam.update(g, optim_stepsize)


    if ckpt_dir is None:
        savedir_fname = tempfile.NamedTemporaryFile().name
    else:
        savedir_fname = osp.join(ckpt_dir, task_name+"_bc")
    U.save_variables(savedir_fname, variables=pi.get_variables())
    return savedir_fname