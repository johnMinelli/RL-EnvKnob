from argparse import Namespace
from copy import deepcopy

import numpy as np
import pygame
import tensorflow as tf

from core.agent import PPO_sol, PPO_gen
from core.model import AC_gen, AC_sol
from keras import backend as K
from env import Skiing

from utils.parser import Options

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
    global parser_config
    args = deepcopy(parser_config)

    # Create the game environment
    env = Skiing(args, fixed_diff=args.eval_difficulty, endless_mode=True)

    # Create the solver and generator agent
    solver, generator = load_agents(args, env.action_space)
    env.set_map_generator(generator)

    # Play the game, using the agent
    play_loop(args, env, solver)


def play_loop(opt: Namespace, env: Skiing, solver: PPO_sol):
    # User input controls
    def register_input(action):
        global user_inactivity
        for event in pygame.event.get():
            if event.type not in [pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.WINDOWLEAVE, pygame.WINDOWENTER, pygame.ACTIVEEVENT]:
                user_inactivity = 0
                if event.type == pygame.QUIT:
                    env.close()
                    exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = np.clip(action-1, a_min=0, a_max=4)
                    if event.key == pygame.K_RIGHT:
                        action = np.clip(action+1, a_min=0, a_max=4)
                    if event.key == pygame.K_KP_PLUS or event.key == pygame.K_p:
                        env.auxiliary_input = max(-1, env.auxiliary_input-0.5)
                    if event.key == pygame.K_KP_MINUS or event.key == pygame.K_m:
                        env.auxiliary_input = min(1, env.auxiliary_input+0.5)
    
                # if event.type == pygame.KEYUP:
                #     if event.key == pygame.K_LEFT:
                #         action = 2
                #     if event.key == pygame.K_RIGHT:
                #         action = 2
        return action
    global user_inactivity
    action = 0
    skip = 2
    user_inactivity = 0
    current_state = env.reset()
    current_state_input, last_state = preprocess(None, current_state, cnn=True)

    while True:
        action = register_input(action)
        if user_inactivity < 100:
            user_inactivity += 1
            skip = 2
        else:
            skip = opt.frame_skipping
            action, action_prob, state_value = solver.take_action(current_state_input)
        next_state, reward, done, info = env.step(action, skip=skip)
        # preprocess new state
        current_state_input, last_state = preprocess(current_state_input, next_state, cnn=True)


if __name__ == '__main__':
    # Get arguments
    global parser_config
    parser_config = Options().parse()

    main()
