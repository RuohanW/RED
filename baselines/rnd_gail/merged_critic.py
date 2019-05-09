import numpy as np
from .rnd_critic import RND_Critic, Enc_Critic
from .mmd_critic import MMD_Critic

def make_critic(env, exp_data, hid_size=128, rnd_hid_size=128, reward_type=0, scale=250000):
    ac_size = env.action_space.sample().shape[0]
    ob_size = env.observation_space.shape[0]
    if reward_type == 0:
        critic = RND_Critic(ob_size, ac_size, hid_size=hid_size, rnd_hid_size=rnd_hid_size, scale=scale)
    elif reward_type == 1:
        critic = Enc_Critic(ob_size, ac_size, hid_size=hid_size, scale=scale)
    else:
        merged_sa = np.concatenate(exp_data, axis=1)
        critic = MMD_Critic(ob_size, ac_size, merged_sa)
    return critic