#!/usr/bin/env python3
import gym
import ext.pia_agent2.ptan as ptan
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

#from ext.pia_agent2.lib import dqn

import os

import retro

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
        #self._decode_discrete_action2 = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)
        # # pkayer 2 : 7: DOWN, 6: 'UP', 15:'BUTTOM'

        # arr = np.array([False] * env.action_space.n)
        # arr[7] = True
        # self._decode_discrete_action2.append(arr)

        # arr = np.array([False] * env.action_space.n)
        # arr[6] = True
        # self._decode_discrete_action2.append(arr)

        # arr = np.array([False] * env.action_space.n)
        # arr[15] = True
        # self._decode_discrete_action2.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

        #print(arr)
        #print(self._decode_discrete_action)

    def action(self, act1):#, act2):
        #print("Actions:", act1, act2)
        act1_v = self._decode_discrete_action[act1].copy()
        #act2_v = self._decode_discrete_action2[act2].copy()
        #print("Action 1 vector: ", act1_v)
        #print("Action 2 vector: ", act2_v)
        return act1_v.copy()#np.logical_or(act1_v, act2_v).copy()

    def step(self, act1):#, act2):
        action = self.action(act1)#, act2)
        #print("Action before step: ", action)
        return self.env.step(action)
        # return self._decode_discrete_action[act].copy()

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()

        print(input_shape)

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
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + (adv - adv.mean(dim=1, keepdim=True))

STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100


def calc_loss(batch, net, tgt_net, gamma, device="cpu", double=True):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    if double:
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default='pong', type=str, help='Choose env')
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--double", default=False, action="store_true", help="Enable double DQN")
    args = parser.parse_args()
    params = common.HYPERPARAMS[args.env]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = retro.make('Pong-Atari2600')
    env = ptan.common.wrappers.wrap_dqn(env)
    env = PongDiscretizer(env)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-double=" + str("True"))

    net = DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0
    eval_states = None
    
    os.makedirs("trained", exist_ok=True)

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < params['replay_initial']:
                continue
            if eval_states is None:
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss_v = calc_loss(batch, net, tgt_net.target_model, gamma=params['gamma'], device=device,
                               double=True)
            loss_v.backward()
            optimizer.step()

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()
            if frame_idx % EVAL_EVERY_FRAME == 0:
                mean_val = calc_values_of_states(eval_states, net, device=device)
                writer.add_scalar("values_mean", mean_val, frame_idx)


    torch.save(net.state_dict(), "trained/RetroPong.pth")

