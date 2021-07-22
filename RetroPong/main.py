####################
### HELMHOLTZ AI ###
####################
# Human vs Pong in openai-retro
# implemented by
# Pia Hanfeld, CASUS
# Nico Hoffmann, HZDR


import pygame
import time
import numpy as np
import retro

########
### IMPORT DQN AGENT & ENV DISCRETIZERS
########
from ext.pia_agent2.agent import Agent, Discretizer, PongDiscretizer
from ext.pia_agent2.rl_hands_on.ptan import ptan


def start_pong(speed_factor = 2):
    # set up controls for player 2
    # if availiable, grab gamepad else fall-back to keyboard (buttons up and down)
    joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
    use_joystick = len(joysticks) > 0 # joystick present?
    if(use_joystick):
        joystick = pygame.joystick.Joystick(0)
        joystick.init()

    # initialize environment
    env = retro.make('Pong-Atari2600', players=2)
    env = ptan.common.wrappers.wrap_dqn(env)
    env = PongDiscretizer(env)
    state = env.reset()
    done = False
    eps_reward = 0
    total_reward = []
    steps = 0



    ## Loading agent
    pia_agent = True
    if (pia_agent):
        agent = Agent(env.action_space.n, path='ext/pia_agent2/RetroPong_reward18.5.pth', nico_arch=False)
    else:
        agent = Agent(env.action_space.n, path='ext/pia_agent2/RetroPong_19.360_newrch.pth',
                       nico_arch=True)

    print("Starting Pong-Atari2600")
    no_games = 0
    while not done:

        # here comes our AI
        p1_action = agent.get_action(state)

        # grab actions from Nimbus gamepad
        # action mapping
        # 0 => down
        # 1 => up
        # 2 => button
        pygame.event.pump()
        if(use_joystick):
            ''' 
            action mapping of our Nimbus controller
            # joystick.get_button(i)
            0 => 1 = button A
            8 => 1 = up
            9 => 1 = down
            10 => 1 = right
            11 => 1 = left
            '''
            actions = [joystick.get_button(9), joystick.get_button(8), 1]
            p2_action = 2 - abs(np.argmax(actions) - 2)

        else:
            # first grab actions of freshly pushed key (event part)
            # then, grab action if key is continuesly being pressed (get_pressed part)
            p2_action = 2
            for event in pygame.event.get():
                if event.type != pygame.KEYDOWN:
                    print(event.type)
                    continue # skip non-keyboard events

                if event.key == pygame.K_UP:
                    p2_action = 1
                if event.key == pygame.K_DOWN:
                    p2_action = 0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                p2_action = 1
            if keys[pygame.K_DOWN]:
                p2_action = 0

        # take a step
        next_state, reward, _, info = env.step(p1_action, p2_action)

        # show game screen
        env.render()
        steps += 1

        # sleep for 100ms as the game might be too fast w/o some sort of delay
        time.sleep(speed_factor * 0.05)

        eps_reward += reward
        if done:
            state = env.reset()
            total_reward.append(eps_reward)
            print("[{}] Total episode reward: {}".format(len(total_reward), eps_reward))
            eps_reward = 0
        else:
            state = next_state

        no_games += 1




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ## init pygame internals
    pygame.init()
    pygame.joystick.init()


    '''
    this code can be used to derive the action mapping of our gamepad
    for i in range(12):
        print(i, joystick.get_button(i))
    '''

    # start playing pong: Human vs AI
    start_pong()