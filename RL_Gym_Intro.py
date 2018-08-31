# import gym
# import numpy as np
# import matplotlib.pyplot as plt
#
# env = gym.make('MountainCar-v0')
# print("Observation space", env.observation_space)
# print("Action space: ", env.action_space)
#
# # plt.imshow(env.render('rgb_array'))
# print("Initial observation:")
# obs0 = env.reset()   # must call reset() before taking a step
#
# print("taking action 2 (right)")
# new_obs, reward, is_done, _ = env.step(2)
#
# print("new observation code:", new_obs)
# print("reward:", reward)
# print("is game over?:", is_done)
#
# # create env manually to set time limit. Please don't change this.
# TIME_LIMIT = 250
# env = gym.wrappers.TimeLimit(gym.envs.classic_control.MountainCarEnv(), max_episode_steps=TIME_LIMIT + 1)
# s = env.reset()
# actions = {'left': 0, 'stop': 1, 'right': 2}
#
# # prepare "display"
# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # fig.show()
#
# def policy(s):
#     if s[0] < 0 and s[1] >= 0:
#         return actions['right']
#     elif s[0] > 0 and s[1] <= 0:
#         return actions['left']
#     else:
#         return actions['stop']
#
# for t in range(TIME_LIMIT):
#
#     # change the line below to reach the flag
#     s, r, done, _ = env.step(policy(s))
#     print(t)
#
#     # draw game image on display
#     # ax.clear()
#     # ax.imshow(env.render('rgb_array'))
#     env.render()
#     # fig.canvas.draw()
#
#     if done:
#         print("Well done!")
#         break
# else:
#     print("Time limit exceeded. Try again.")
# env.close()

