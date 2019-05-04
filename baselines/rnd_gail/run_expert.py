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

from baselines.run import get_exp_data


Log_dir = osp.expanduser("~/workspace/log/mujoco")
Checkpoint_dir = osp.expanduser("~/workspace/checkpoint/mujoco/")


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v2')
    parser.add_argument('--pi', help="model file", type=str, default='trpo.gamma_0.990000.Hopper.seed_0.reward_2kl_0.01g_3_1')
    parser.add_argument('--bc', help='BC policy', default=0, type=int)
    parser.add_argument('--uid', help='timestamp_id', default='', type=str)
    parser.add_argument('--render', help='Save to video', default=0, type=int)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='data/deterministic.trpo.Hopper.0.00.npz')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default=Checkpoint_dir)
    parser.add_argument('--log_dir', help='the directory to save log file', default=Log_dir)
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='evaluate')
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=2e6)
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=20)
    parser.add_argument('--gamma', help='Discount factor', type=float, default=0.97)
    boolean_flag(parser, 'popart', default=True, help='Use popart on V function')
    return parser.parse_args()


def get_task_name(args):
    task_name = args.algo+"."
    if args.pretrained:
        task_name += "pretrained."
    task_name +="%f." % args.gamma
    task_name += args.env_id.split("-")[0]
    task_name = task_name + ".g_step_" + str(args.g_step) + ".d_step_" + str(args.d_step)
    task_name += ".seed_" + str(args.seed)
    return task_name


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
    task_name = get_task_name(args) + args.uid
    if args.bc:
        task_name+="_bc"


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

    if args.task == 'train':
        logger.configure(dir=args.log_dir, log_suffix=task_name, format_strs=["log", "stdout"])
        exp_data = get_exp_data(osp.join(osp.dirname(osp.realpath(__file__)), "../../data/%s.pkl" % args.env_id))
        critic = make_critic(env, exp_data).rnd
        train(env,
              args.seed,
              policy_fn,
              critic,
              exp_data,
              args.algo,
              args.g_step,
              args.d_step,
              args.policy_entcoeff,
              args.num_timesteps,
              args.save_per_iter,
              args.checkpoint_dir,
              args.pretrained,
              args.BC_max_iter,
              args.gamma,
              task_name
              )
    elif args.task == 'evaluate':
        # load_path = osp.join(Checkpoint_dir, task_name)
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
               save=args.save_sample
               )
    else:
        raise NotImplementedError
    env.close()


def train(env, seed, policy_fn, reward_giver, dataset, algo,
          g_step, d_step, policy_entcoeff, num_timesteps, save_per_iter,
          checkpoint_dir, pretrained, BC_max_iter, gamma, task_name=None):

    pretrained_weight = None
    if pretrained and (BC_max_iter > 0):
        # Pretrain with behavior cloning
        from baselines.rnd_gail.behavior_clone import learn as bc_learn
        pretrained_weight = bc_learn(env, policy_fn, dataset, task_name, max_iters=BC_max_iter, ckpt_dir=checkpoint_dir)

    if algo == 'trpo':
        from baselines.rnd_gail import trpo_mpi
        # Set up for MPI seed
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        trpo_mpi.learn(env, policy_fn, reward_giver, dataset, rank,
                       pretrained=pretrained, pretrained_weight=pretrained_weight,
                       g_step=g_step, d_step=d_step,
                       entcoeff=policy_entcoeff,
                       max_timesteps=num_timesteps,
                       ckpt_dir=checkpoint_dir,
                       save_per_iter=save_per_iter,
                       timesteps_per_batch=1024,
                       max_kl=0.01, cg_iters=10, cg_damping=0.1,
                       gamma=gamma, lam=0.97,
                       vf_iters=5, vf_stepsize=1e-3,
                       task_name=task_name)
    else:
        raise NotImplementedError


def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False):

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
    if save:
        filename = load_model_path.split('/')[-1] + '.' + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))
    avg_len = np.mean(len_list)
    avg_ret = np.mean(ret_list)
    print(ret_list)
    # print("Average length:", avg_len)
    # print("Average return:", avg_ret)
    # print("std:", np.std(ret_list))
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
