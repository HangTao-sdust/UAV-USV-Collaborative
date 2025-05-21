# import argparse
# import torch
# import time
# import imageio
# import numpy as np
# from pathlib import Path
# from torch.autograd import Variable
# from utils.make_env import make_env
# from algorithms.maddpg import MADDPG
#
#
# def run(config):
#     model_path = (Path('./models') / config.env_id / config.model_name /
#                   ('run%i' % config.run_num))
#     if config.incremental is not None:
#         model_path = model_path / 'incremental' / ('model_ep%i.pt' %
#                                                    config.incremental)
#     else:
#         model_path = model_path / 'model.pt'
#
#     if config.save_gifs:
#         gif_path = model_path.parent / 'gifs'
#         gif_path.mkdir(exist_ok=True)
#
#     maddpg = MADDPG.init_from_save(model_path)
#     env = make_env(config.env_id, discrete_action=maddpg.discrete_action)
#     maddpg.prep_rollouts(device='cpu')
#     ifi = 1 / config.fps  # inter-frame interval
#
#     for ep_i in range(config.n_episodes):
#         print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
#         obs = env.reset()
#         if config.save_gifs:
#             frames = []
#             frames.append(env.render('rgb_array')[0])
#         env.render('human')
#         for t_i in range(config.episode_length):
#             calc_start = time.time()
#             # rearrange observations to be per agent, and convert to torch Variable
#             torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
#                                   requires_grad=False)
#                          for i in range(maddpg.nagents)]
#             # get actions as torch Variables
#             torch_actions = maddpg.step(torch_obs, explore=False)
#             # convert actions to numpy arrays
#             actions = [ac.data.numpy().flatten() for ac in torch_actions]
#             obs, rewards, dones, infos = env.step(actions)
#             if config.save_gifs:
#                 frames.append(env.render('rgb_array')[0])
#             calc_end = time.time()
#             elapsed = calc_end - calc_start
#             if elapsed < ifi:
#                 time.sleep(ifi - elapsed)
#             env.render('human')
#         if config.save_gifs:
#             gif_num = 0
#             while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
#                 gif_num += 1
#             imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
#                             frames, duration=ifi)
#
#     env.close()
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--env_id", default="mysimple2", help="Name of environment")
#     parser.add_argument("--model_name", default="mymodel",
#                         help="Name of model")
#     parser.add_argument("--run_num", default=4, type=int)
#     parser.add_argument("--save_gifs", action="store_true",
#                         help="Saves gif of each episode into model directory")
#     parser.add_argument("--incremental", default=None, type=int,
#                         help="Load incremental policy from given episode " +
#                              "rather than final policy")
#     parser.add_argument("--n_episodes", default=100, type=int)
#     parser.add_argument("--episode_length", default=50, type=int)
#     parser.add_argument("--fps", default=30, type=int)
#
#     config = parser.parse_args()
#
#     run(config)
import argparse
import torch
import time
import imageio
import numpy as np
import matplotlib.pyplot as plt  # 引入matplotlib库用于绘图
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG

response_time_uav = [[], [], [], [], [], []]
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

    maddpg = MADDPG.init_from_save(model_path)
    env = make_env(config.env_id, discrete_action=maddpg.discrete_action)
    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval

    total_rewards = []  # 记录每个episode的总奖励

    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        total_episode_reward = 0  # 记录当前episode的总奖励
        if config.save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])
        env.render('human')
        for t_i in range(config.episode_length):
            print(f"step {t_i}")
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_actions = maddpg.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            obs, rewards, dones, infos = env.step(actions)
            # print(rewards)
            total_episode_reward += (sum(rewards)/3)  # 累加当前step的奖励
            if config.save_gifs:
                frames.append(env.render('rgb_array')[0])
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            env.render('human')
        total_rewards.append(total_episode_reward)  # 保存当前episode的总奖励
        if config.save_gifs:
            gif_num = 0
            while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                gif_num += 1
            imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
                            frames, duration=ifi)
        ii = 0
        for agent in env.agents:
            if agent.type == 'UAV':
                print(f"{agent.type} {agent.name} 响应时间:{agent.response_time}")
                response_time_uav[ii].append(agent.response_time)
                ii += 1
    env.close()
    averages_response = []
    for row in response_time_uav:
        modified_row = [50 if x == -1.0 else x for x in row]
        average = sum(modified_row) / len(modified_row)
        averages_response.append(average)
    print(response_time_uav)
    print(averages_response)
    print(f"ave:{sum(averages_response)/len(averages_response)}")
    # 绘制奖励曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(config.n_episodes), total_rewards, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode during Training")
    plt.grid(True)
    plt.legend()
    plt.show()  # 展示奖励曲线


if __name__ == '__main__':
    import random
    torch.manual_seed(2)
    np.random.seed(2)
    random.seed(2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="mysimple2", help="Name of environment")
    parser.add_argument("--model_name", default="mymodel",
                        help="Name of model")
    parser.add_argument("--run_num", default=9, type=int)
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=1009, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument("--episode_length", default=50, type=int)
    parser.add_argument("--fps", default=30, type=int)

    config = parser.parse_args()

    run(config)
