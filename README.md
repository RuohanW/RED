# RED: Random Expert Distillation

This is the implementation for the paper **Random Expert Distillation: Imitation Learning via Expert Policy Support Estimation** from ICML 2019.

RED leverages the Trust Region Policy Policy Optimization (TRPO) implementation from OpenAI's [baselines](https://github.com/openai/baselines). Please refer to the baselines repo for installation prerequisites and instructions.

## Models
We provide implementation of three models, RND, AE and GMMIL in the rnd_gail/ folder. They correspond with reward type 0, 1 and 2.

## Training
To train a model:
```bash
python rnd_gail/mujoco_main.py --env_id=<environment_id> --reward=<reward_model> [additional arguments]
```
We have provided a working configuration of hyper parameters in rnd_gail/mujoco_main for Mujoco tasks. To override them from the command line, please disable the defaults in the script first.

### Example: RND with MuJoCo Hopper
For instance, to train MuJoCo Hopper using RND for 2M timesteps
```bash
python rnd_gail/mujoco_main.py --env_id=Hopper-v2 --reward=0 --num_timesteps=2e6
```

## Saving and loading models
Models are saved at <user_home>/workspace/checkpoint/mujoco/.
To run a saved model:
```bash
python rnd_gail/run_expert.py --env_id=<environment_id> --pi=<model_filename>
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

