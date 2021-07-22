####################
### HELMHOLTZ AI ###
####################
# Human vs Pong in openai-retro
# implemented by
# Pia Hanfeld, CASUS
# Nico Hoffmann, HZDR

import numpy as np
import retro

import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

import gym

from collections import deque

from time import sleep

import ext.pia_agent2.rl_hands_on.ptan.ptan as ptan
import cv2


def PongDiscretizer(env):
    """
    Discretize Retro Pong-Atari2600 environment
    """
    return Discretizer(env, buttons=env.unwrapped.buttons, combos=[['DOWN'], ['UP'], ['BUTTON'], ])


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    Args:
        buttons: ordered list of buttons, corresponding to each dimension of the MultiBinary action space
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, buttons, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        self._decode_discrete_action = []
        self._decode_discrete_action2 = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)
        # # pkayer 2 : 7: DOWN, 6: 'UP', 15:'BUTTOM'

        arr = np.array([False] * env.action_space.n)
        arr[7] = True
        self._decode_discrete_action2.append(arr)

        arr = np.array([False] * env.action_space.n)
        arr[6] = True
        self._decode_discrete_action2.append(arr)

        arr = np.array([False] * env.action_space.n)
        arr[15] = True
        self._decode_discrete_action2.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

        #print(arr)
        #print(self._decode_discrete_action)

    def action(self, act1, act2):
        #print("Actions:", act1, act2)
        act1_v = self._decode_discrete_action[act1].copy()
        act2_v = self._decode_discrete_action2[act2].copy()
        #print("Action 1 vector: ", act1_v)
        #print("Action 2 vector: ", act2_v)
        return np.logical_or(act1_v, act2_v).copy()

    def step(self, act1, act2):
        action = self.action(act1, act2)
        #print("Action before step: ", action)
        return self.env.step(action)
        # return self._decode_discrete_action[act].copy()


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions, nico_arch):
        super(DuelingDQN, self).__init__()

        #print(input_shape)

        if(nico_arch == True):
            self.conv = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=4, stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.LeakyReLU()
            )

            conv_out_size = self._get_conv_out(input_shape)

            self.fc_adv = nn.Sequential(
                nn.Linear(conv_out_size, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 256),
                nn.LeakyReLU(),
                nn.Linear(256, n_actions)
            )
            self.fc_val = nn.Sequential(
                nn.Linear(conv_out_size, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 1)
            )

        else:
            self.conv = nn.Sequential(
               nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
               nn.ReLU(),
               nn.Conv2d(32, 64, kernel_size=4, stride=2),
               nn.ReLU(),
               nn.Conv2d(64, 64, kernel_size=3, stride=1),
               nn.ReLU()
            )

            conv_out_size = self._get_conv_out(input_shape)

            self.fc_adv = nn.Sequential(
                nn.Linear(conv_out_size, 256),
                nn.ReLU(),
                nn.Linear(256, n_actions)
            )
            self.fc_val = nn.Sequential(
                nn.Linear(conv_out_size, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256.
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class Agent():
    """
    An agent for the Pong environment. It loads the stored DQN agent, preprocesses the states returned by the environment, 
    and returns the predicted actions by the DQN.
    Args:
        n_actions: Integer, amount of possible actions of the retro environment
        path: String, path to the stored DQN
    """
    def __init__(self, n_actions, path, nico_arch = False):
        
        self.n_actions = n_actions
        self.transition_memory = deque(maxlen=4)                                        # container holding the past 4 preprocessed states
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    # specifies the device on which the DQN and the inputs/outputs are stored
        
        self.dqn = DuelingDQN(n_actions=3, input_shape=[4, 84, 84], nico_arch = nico_arch)                     # initilize the DQN,
        self.dqn.load_state_dict(torch.load(path, map_location=self.device))            # load the saved model at the specified location
        self.dqn.eval()                                                                 # and freeze it

        # actions for retro
        # 0 => down
        # 1 => up
        # 2 => button

        #self.action_map = [2, 2, 1, 0, 1, 0]

        #self.action_counter = 0

    def process(self, frame):
        img = np.reshape(frame, [210, 160, 3]).astype(np.float32)                   # make sure frame is in the right shape,
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114    # grayscale,
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)   # crop,
        x_t = resized_screen[18:102, :]                                             # reduce to single channel,
        x_t = np.reshape(x_t, [84, 84, 1])                                          # make sure frame is in the right shape
        return x_t.astype(np.uint8)

    def get_action(self, state):
        """
        Map and return the predicted action by the DQN.
        Args:
            state: the current (210, 160, 3) state returned by the environment
        Returns:
            an integer between [0, 2]
        """
        #self.transition_memory.append(self.process(state))                                   # append the current state to memory of past states

        #if len(self.transition_memory) < 4:                                                     # if the memory of past states is not sufficiently filled
        #    return np.random.randint(self.n_actions)                                            # return a random action

        #transition = torch.FloatTensor(list(self.transition_memory))                             # turn deque into tensor
        #transition = transition.squeeze(3).unsqueeze(0).to(self.device)                     # ensure correct shape and send to correct device
        #action = torch.argmax(self.dqn(transition))                                         # get best action for current transition
        #print("Predicted action: ", action)
        state = torch.tensor(state.__array__()).unsqueeze(0).to(self.device)
        action = torch.argmax(self.dqn(state))
        return action.item()
        #print("Mapped action: ", self.action_map[action])
       # return self.action_map[action]                                                          # return the mapped action
        #action_map = [0, 1, 2, 3, 2, 3]
        #return action_map[action]
        #return action



def main():
    env = retro.make('Pong-Atari2600', players=2)
    env = ptan.common.wrappers.wrap_dqn(env)
    env = PongDiscretizer(env)

    state = env.reset()
    done = False
    eps_reward = 0
    total_reward = []


    agent = Agent(env.action_space.n, path='agents/RetroPong_reward18.5.pth')

    while not done:
        p1_action = agent.get_action(state)
        p2_action = env.action_space.sample()
        print(p1_action)#, p2_action)
        #if p1_action == past_action:
        #    action_counter += 1#

        #if action_counter > 15:
        #    p1_action = 2
        #    action_counter = 0
        
        next_state, reward, done, _ = env.step(p1_action, p2_action)

        #if render:s
        env.render()
        #sleep(0.01)
        print(reward)
        eps_reward += reward
        if done:
            state = env.reset()
            total_reward.append(eps_reward)
            print("[{}] Total episode reward: {}".format(len(total_reward), eps_reward))
            eps_reward = 0
        else:
            state = next_state
            past_action = p1_action

if __name__ == "__main__":
    main()