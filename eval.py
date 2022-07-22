from argparse import Namespace
from copy import deepcopy

import numpy as np
import pandas as pd
import pygame
import wandb
import tensorflow as tf
from tensorboardX import SummaryWriter

from core.agent import PPO_sol, PPO_gen
from core.model import AC_gen, AC_sol
from keras import backend as K, Model
from core.env import Skiing

from utils.logger import Logger
from utils.parser import Options
from utils.utils import fix_random

EVAL_EPISODES = 30

def load_agents(opt: Namespace, action_space: int):
    """
    Creates the atari skiing agent.

    :return: the agent.
    """

    # create solver network
    solver_model = AC_sol(opt, action_space, opt.load_path, opt.load_solver_agent)
    # create the solver.
    solver = PPO_sol(opt, solver_model, frozen=False)

    if opt.generator:
        # create generator network
        generator_model = AC_gen(opt, action_space, opt.load_path, opt.load_generator_agent) if opt.generator else None
        # create the generator.
        generator = PPO_gen(opt, generator_model, frozen=True)
    else:
        generator = None

    return solver, generator


def preprocess(stacked_states: tf.Tensor, new_state: np.ndarray, cnn: bool):
    if cnn:
        new_state = K.expand_dims(tf.convert_to_tensor(new_state, tf.float32), -1)
        if stacked_states is None:
            return K.concatenate([new_state, new_state, new_state, new_state]), new_state
        else:
            return K.concatenate([new_state, stacked_states[..., :-1]]), new_state
    else:
        return tf.convert_to_tensor(new_state, tf.float32)


def main():
    global best_result, parser_config
    args = deepcopy(parser_config)

    # Init loggers
    if args.wandb:
        wandb.init(project="EnvKnob", entity="johnminelli")
    if args.tensorboard:
        tb_writer = SummaryWriter()
    else: tb_writer = None
    logger = Logger(mode="eval", prefix="eval", episodes=EVAL_EPISODES, batch_size=args.batch_size, terminal_print_freq=args.print_freq, tensorboard=tb_writer, wand=args.wandb)

    # Set the seed
    # fix_random(args.seed)

    # Create the game environment
    env = Skiing(args, fixed_diff=args.eval_difficulty)

    # Create the solver and generator agent
    solver, generator = load_agents(args, env.action_space)
    env.set_map_generator(generator)

    # Play the game, using the agent
    play_loop(args, env, solver, logger)


def play_loop(opt: Namespace, env: Skiing, solver: PPO_sol, logger: Logger):
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

    for episode in range(EVAL_EPISODES):
        current_state = env.reset()
        current_state_input, last_state = preprocess(None, current_state, cnn=True)
        logger.episode_start(episode)
        done = False
        total_reward = 0.0
        step = 0

        while not done:
            action, action_prob, state_value = solver.take_action(current_state_input)
            next_state, reward, done, info = env.step(action, skip=opt.frame_skipping)
            total_reward += reward
            logger.step(step, {}, 0)

            # preprocess new state
            current_state_input, last_state = preprocess(current_state_input, next_state, cnn=True)

            step += 1
            not_running = not register_input()
            if done or not_running:
                break

        logger.episode_stop(total_reward, env.score_points)

        # if not_running: break
    env.close()


if __name__ == '__main__':
    # Get arguments
    global parser_config
    parser_config = Options().parse()

    main()
