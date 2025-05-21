import argparse
import time
from pathlib import Path
import multiprocessing
import imageio
import torch
from torch.autograd import Variable
from algorithms.attention_sac import AttentionSAC
from utils.make_env import make_env
import numpy as np


def run(config):
    model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'model.pt'

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(exist_ok=True)

    maac = AttentionSAC.init_from_save(model_path)
    env = make_env(config.env_id,discrete_action=True,is_train=config.is_train)
    maac.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval

    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        if config.save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])
        env.render('human')

        for t_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maac.nagents)]
            # get actions as torch Variables
            torch_actions = maac.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            obs, rewards, dones, infos = env.step(actions)
            if config.save_gifs:
                frames.append(env.render('rgb_array')[0])
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            env.render('human')
        if config.save_gifs:
            gif_num = 0
            while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                gif_num += 1
            imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
                            frames, duration=ifi)

    env.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="uav_usv_search",default="uav_usv_search")
    parser.add_argument("model_name",
                        help="Name of model",default="model_1")
    parser.add_argument("run_num", default=6, type=int)
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=1, type=int)
    parser.add_argument("--episode_length", default=300, type=int)
    parser.add_argument("--fps", default=50, type=int)
    parser.add_argument("--is_train", default=False, type=bool)
    config = parser.parse_args()

    run(config)


