'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

import argparse
import os.path as osp
import logging
from mpi4py import MPI
from tqdm import tqdm

import numpy as np
import gym

from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger
from baselines.rnd_gail.merged_critic import make_critic

from baselines.rnd_gail.mujoco_main import get_exp_data


Log_dir = osp.expanduser("~/workspace/log/mujoco")
Checkpoint_dir = osp.expanduser("~/workspace/checkpoint/mujoco/")


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v2')
    parser.add_argument('--pi', help="model file", type=str, default='')
    parser.add_argument('--render', help='Save to video', default=0, type=int)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default=Checkpoint_dir)
    parser.add_argument('--log_dir', help='the directory to save log file', default=Log_dir)
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    boolean_flag(parser, 'popart', default=True, help='Use popart on V function')
    return parser.parse_args()

def main(args):
    # set_global_seeds(args.seed)
    env = gym.make(args.env_id)
    if args.render:
        vid_dir = osp.expanduser("~/Videos")

        env.env._get_viewer("rgb_array")
        env.env.viewer_setup()

        env = gym.wrappers.Monitor(env, vid_dir, video_callable=lambda ep: True, force=True, mode="evaluation")

    # env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    if "gail" in args.pi:
        from baselines.gail import mlp_policy
        def policy_fn(name, ob_space, ac_space, reuse=False):
            return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                        reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)
    else:
        from baselines.rnd_gail import mlp_policy
        def policy_fn(name, ob_space, ac_space,):
            return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                        hid_size=args.policy_hidden_size, num_hid_layers=2, popart=args.popart)

    if args.log_dir != Log_dir:
        log_dir = osp.join(Log_dir, args.log_dir)
        load_dir = osp.join(Checkpoint_dir, args.log_dir)
    else:
        load_dir = Checkpoint_dir
        log_dir = Log_dir
    logger.configure(dir=log_dir, log_suffix="eval" + args.pi, format_strs=["log", "stdout"])
    load_path = osp.join(load_dir, args.pi)
    runner(env,
           policy_fn,
           load_path,
           timesteps_per_batch=1000,
           number_trajs=50,
           stochastic_policy=args.stochastic_policy,
           )
    env.close()


def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy):

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    pi.load_policy(load_model_path)

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    for _ in tqdm(range(number_trajs)):
        traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        obs_list.append(obs)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)
    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    avg_len = np.mean(len_list)
    avg_ret = np.mean(ret_list)
    print(ret_list)
    logger.info(avg_len)
    logger.info(avg_ret)
    logger.info(np.std(ret_list))
    return avg_len, avg_ret


# Sample one trajectory (until trajectory end)
def traj_1_generator(pi, env, horizon, stochastic):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []

    while True:
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
    return traj


if __name__ == '__main__':
    args = argsparser()
    main(args)
