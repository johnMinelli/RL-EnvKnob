from argparse import Namespace
from copy import deepcopy

import numpy as np
import pandas as pd
import pygame
import wandb
import tensorflow as tf
from tensorboardX import SummaryWriter

from core.agent import GAE_PPO
from core.model import initialize_optimizer, AC
from keras import backend as K, Model
from env import Skiing

from utils.logger import Logger
from utils.parser import Options
from utils.utils import fix_random


def create_agents(opt: Namespace, action_space: int):
    """
    Creates the atari skiing agent.

    :return: the agent.
    """

    # create policy network
    policy_model = AC(opt, action_space, opt.load_path, opt.load_solver_agent)
    # create generator network
    generator_model = AC(opt, action_space, opt.load_path, opt.load_generator_agent)
    # Create the solver.
    solver = GAE_PPO(opt, policy_model, frozen=False)
    # Create the solver.
    generator = GAE_PPO(opt, generator_model, frozen=True)

    return solver, generator


def preprocess(stacked_states: tf.Tensor, new_state: np.ndarray, cnn: bool):
    if cnn:
        new_state = K.expand_dims(tf.convert_to_tensor(new_state, tf.float32), -1)
        if stacked_states is None:
            return K.concatenate([new_state, new_state, new_state, new_state])
        else:
            return K.concatenate([new_state, stacked_states[..., :-1]])
    else:
        return tf.convert_to_tensor(new_state, tf.float32)


def play_loop(opt: Namespace, env: Skiing, solver: Model, generator: Model, logger: Logger):
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

    action = 0
    not_running = False
    ppo_batch = opt.batch_size
    trained_agent = solver
    for episode in range(opt.episodes):
        current_state = env.reset()
        current_state_input = preprocess(None, current_state, opt.cnn>0)
        logger.episode_start(episode)
        done = False
        total_reward = 0.0

        while not done:
            action_probs, state_value = solver.take_action(current_state_input)
            action = np.random.choice(env.action_space, p=action_probs[0, :].numpy())
            next_state, reward, done, info = env.step(action, skip=opt.frame_skipping)

            trained_agent.fill_buffer(current_state_input, state_value, action_probs[0, action], action, reward, int(done))

            # motion interval: can be increased by changing env scroll speed
            current_state_input = preprocess(current_state_input, next_state, opt.cnn>0)
            total_reward += reward

            not_running = not register_input()
            if done or not_running:
                break

        # add last state value estimate needed for advantage computing
        _, state_value = solver.take_action(current_state_input)
        trained_agent.buffer.state_values.append(state_value)
        # update with data batch collected
        trained_agent.update(logger)

        logger.episode_stop(total_reward, env.score_points)

        # if episode != 0 and (episode % opt.alternate_training_interval == 0):
        #     solver.freeze_switch()
        #     generator.freeze_switch()
        #     trained_agent = generator if solver.is_frozen else solver

        if episode != 0 and (episode % opt.agent_save_interval == 0):
            print(f'Saving agent at episode: {episode}.')
            solver.save_model(episode)

        if not_running: break
    env.close()


def main():
    global best_result, parser_config
    args = deepcopy(parser_config)

    # Init loggers
    if args.wandb:
        wandb.init(project="EnvKnob", entity="johnminelli")
        # Init sweep agent configuration
        if args.sweep_id is not None: args.__dict__.update(wandb.config)
        wandb.config = args
        wandb.log({"params": wandb.Table(data=pd.DataFrame({k: [v] for k, v in vars(args).items()}))})
    if args.tensorboard:
        tb_writer = SummaryWriter()
    else: tb_writer = None
    train_logger = Logger(mode="train", episodes=args.episodes, batch_size=args.batch_size, terminal_print_freq=args.print_freq, tensorboard=tb_writer, wand=args.wandb)

    # Set the seed
    fix_random(args.seed)

    # Create the game environment
    env = Skiing(args, None)

    # Create the solver and generator agent
    solver, generator = create_agents(args, env.action_space)
    # env.set_map_generator(generator.take_action)

    # Play the game, using the agent
    play_loop(args, env, solver, generator, train_logger)


if __name__ == '__main__':
    # Get arguments
    global parser_config
    parser_config = Options().parse()

    if parser_config.sweep_id is not None:
        wandb.agent(parser_config.sweep_id, main)
    else:
        main()

    # To get a SWEEP ID:
    #   sweep_configuration = {
    #       "name": "my-awesome-sweep", "metric": {"name": "accuracy", "goal": "maximize"}, "method": "grid",
    #       "parameters": {"a": {"values": [1, 2, 3, 4]}}
    #   }
    #   print(wandb.sweep(sweep_configuration))
    #
    # Or from CommandLine:
    #   wandb sweep config.yaml
    #
    # Or from web interface

