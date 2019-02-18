import gym
import coding_challenge
import numpy as np

env = gym.make('Battleship-v0')
state = env.reset()
terminal = False
while not terminal:
    action = np.random.rand(2)
    state, reward, terminal, info = env.step(action)
    print(info['game_message'])
