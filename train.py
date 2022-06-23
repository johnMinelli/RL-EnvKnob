from argparse import Namespace

import gym
import numpy as np
import pygame
import wandb
from tensorboardX import SummaryWriter

from core.agent import DQN, load_dqn_agent, GAE_PPO
from core.model import initialize_optimizer, AC
from keras import backend as K
from env import Skiing
from game_engine.game import Game

from utils.logger import Logger
from utils.parser import Options
from utils.utils import fix_random


class IncompatibleAgentConfigurationError(Exception):
    pass


def create_agents(opt: Namespace, action_space: int):
    """
    Creates the atari skiing agent.

    :return: the agent.
    """
    if opt.agent != '':
        pass
        # # Load the agent.
        # dqn = load_dqn_agent(opt.agent, {'huber_loss': huber_loss})
        # 
        # # Check for agent configuration conflicts.
        # if dqn.observation_space_shape != game.observation_space_shape:
        #     raise IncompatibleAgentConfigurationError('Incompatible observation space shapes have been encountered.'
        #                                               'The loaded agent has shape {}, '
        #                                               'but the new requested shape is {}.'
        #                                               .format(dqn.observation_space_shape,
        #                                                       game.observation_space_shape))
        # 
        # if dqn.action_size != game.action_space_size:
        #     raise IncompatibleAgentConfigurationError('')
        # 
        # # Use the new configuration parameters.
        # dqn.target_model_change = opt.target_change_interval
        # dqn.gamma = opt.gamma
        # dqn.batch_size = opt.batch_size
        # dqn.policy = policy
        # print('Agent {} has been loaded successfully.'.format(opt.agent))
    else:
        # create policy network
        policy_model = AC(opt, action_space)
        # create generator network
        generator_model = AC(opt, action_space)
        # Create the solver.
        solver = GAE_PPO(opt, policy_model, frozen=False)
        # Create the solver.
        generator = GAE_PPO(opt, generator_model, frozen=True)

    return solver, generator


# def prepro(I):  # TODO appropriate preprocessing
#   """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
#   I = I[35:195] # crop
#   I = I[::2,::2,0] # downsample by factor of 2
#   I[I == 144] = 0 # erase background (background type 1)
#   I[I == 109] = 0 # erase background (background type 2)
#   I[I != 0] = 1 # everything else (paddles, ball) just set to 1
#   return I.astype(np.float).ravel()
# cur_x = prepro(observation)
# x = cur_x - prev_x if prev_x is not None else np.zeros(D)
# prev_x = cur_x
# 


def play_loop(opt: Namespace, env: gym.Env, solver, generator):
    # User input controls
    def register_input():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = np.clip(action-1, a_min=0, a_max=4)
                if event.key == pygame.K_RIGHT:
                    action = np.clip(action+1, a_min=0, a_max=4)
            # if event.type == pygame.KEYUP:
            #     if event.key == pygame.K_LEFT:
            #         action = 2
            #     if event.key == pygame.K_RIGHT:
            #         action = 2
        return True

    # Init loggers
    if opt.tensorboard:
        tb_writer = SummaryWriter()
    else: tb_writer = None
    if opt.wandb:
        wandb.init(project="EnvKnob", entity="johnminelli")

    train_logger = Logger(mode="train", episodes=opt.episodes, batch_size=opt.batch_size, terminal_print_freq=opt.print_freq, tensorboard=tb_writer, wand=args.wandb)

    action = 0
    not_running = False
    ppo_batch = opt.batch_size
    trained_agent = solver
    for episode in range(opt.episodes):
        current_state = current_state_input = env.reset()
        train_logger.episode_start(episode)
        done = False
        total_reward = 0.0

        while not done:
            action_prob, state_value = solver.take_action(current_state_input)
            action = np.random.choice(env.action_space, p=action_prob[0, :])
            next_state, reward, done, info = env.step(action, skip=opt.frame_skipping)

            trained_agent.fill_buffer(current_state_input, state_value, action_prob, reward, done)

            # motion interval: can be increased by changing env scroll speed
            current_state_input = next_state  # - current_state  # TODO maybe?
            current_state = next_state
            total_reward += reward

            if done or (not_running := not register_input()):
                break

        # add last state value estimate needed for advantage computing
        _, state_value = solver.take_action(current_state_input)
        trained_agent.buffer.state_values.append(state_value)
        # update with mini batch collected
        losses = trained_agent.update(train_logger)

        avg_time, avg_loss, avg_reward = train_logger.episode_stop(total_reward, env.steps)

        if episode != 0 and (episode % opt.alternate_training_interval):
            solver.freeze_switch()
            generator.freeze_switch()
            trained_agent = generator if solver.is_frozen else solver

        if episode % opt.agent_save_interval == 0:
            print('Saving agent.')
            # filename = solver.save_agent(str(episode))
            # print('Agent has been successfully saved as {}.'.format(filename))

        if not_running: break
    env.close()


if __name__ == '__main__':
    # Get arguments
    args = Options().parse()
    fix_random(args.seed)

    # Create the game environment
    env = Skiing(args, None)

    # Create the solver and generator agent
    solver, generator = create_agents(args, env.action_space)
    # env.set_map_generator(generator.take_action)
    # Play the game, using the agent
    play_loop(args, env, solver, generator)
