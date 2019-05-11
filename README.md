# RED: Random Expert Distillation

This is the implementation for the paper **Random Expert Distillation: Imitation Learning via Expert Policy Support Estimation** from ICML 2019.
This repo is part of the software offered by [Personal Robotics Lab@Imperial](https://github.com/orgs/ImperialCollegeLondon/teams/personal-robotics-lab/repositories).

RED leverages the Trust Region Policy Policy Optimization (TRPO) implementation from OpenAI's [baselines](https://github.com/openai/baselines). Please refer to the baselines repo for installation prerequisites and instructions.

## Models
We provide implementation of three models in `rnd_gail/folder`. They correspond to command line argument `--reward=` 0, 1 and 2.
1. **Random Expert Distillation (RED)**: reward function from expert support estimation with random prediction problems.
2. **AutoEncoder (AE)**: reward function from expert support estimation with autoencoder prediction.
3. **Generative Moment Matching Imitation Learning (GMMIL)**: benchmark method from [this work](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16807/16720).

## Training
To train a model:
```bash
$ python rnd_gail/mujoco_main.py --env_id=<environment_id> --reward=<reward_model> [additional arguments]
```
We have provided a working configuration of hyper parameters in `rnd_gail/mujoco_main` for Mujoco tasks. To override them from the command line, please disable the defaults in the script first.

### Example: RND with MuJoCo Hopper
For instance, to train MuJoCo Hopper using RED for 2M timesteps
```bash
$ python rnd_gail/mujoco_main.py --env_id=Hopper-v2 --reward=0 --num_timesteps=2e6
```

## Saving and loading models
Models are saved at `<user_home>/workspace/checkpoint/mujoco/`.
To run a saved model:
```bash
$ python rnd_gail/run_expert.py --env_id=<environment_id> --pi=<model_filename>
```

## Reference
To cite this work please refer to:

    @inproceedings{wang2019random,
      author = {Wang, Ruohan and Ciliberto, Carlo and Amadori, Pierlugi and Demiris, Yiannis},
      title = {Random Expert Distillation: Imitation Learning via Expert Policy Support Estimation},
      year = {2019},
      booktitle = {Proceedings of International Conference on Machine Learning},
      organization = {ACM},
    }

