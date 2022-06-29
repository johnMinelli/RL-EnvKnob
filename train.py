from argparse import Namespace
from copy import deepcopy

import numpy as np
import pandas as pd
import pygame
import wandb
import tensorflow as tf
from tensorboardX import SummaryWriter

from core.agent import PPO, PPO_sol, PPO_gen
from core.model import initialize_optimizer, AC, AC_gen, AC_sol
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

    # create solver network
    solver_model = AC_sol(opt, action_space, opt.load_path, opt.load_solver_agent)
    # create the solver.
    solver = PPO_sol(opt, solver_model, frozen=False)

    if opt.generator_train:
        # create generator network
        generator_model = AC_gen(opt, action_space, opt.load_path, opt.load_generator_agent) if opt.generator_train else None
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
        # Init sweep agent configuration
        if args.sweep_id is not None: args.__dict__.update(wandb.config)
        wandb.config = args
        wandb.log({"params": wandb.Table(data=pd.DataFrame({k: [v] for k, v in vars(args).items()}))})
    if args.tensorboard:
        tb_writer = SummaryWriter()
    else: tb_writer = None
    sol_logger = Logger(mode="train", prefix="sol", episodes=args.episodes, batch_size=args.batch_size, terminal_print_freq=args.print_freq, tensorboard=tb_writer, wand=args.wandb)
    gen_logger = Logger(mode="train", prefix="gen", episodes=args.episodes, batch_size=args.batch_size, terminal_print_freq=args.print_freq, tensorboard=tb_writer, wand=args.wandb)

    # Set the seed
    fix_random(args.seed)

    # Create the game environment
    env = Skiing(args)

    # Create the solver and generator agent
    solver, generator = create_agents(args, env.action_space)
    env.set_map_generator(generator)

    # Play the game, using the agent
    play_loop(args, env, solver, generator, sol_logger, gen_logger)


def play_loop(opt: Namespace, env: Skiing, solver: PPO_sol, generator: PPO_gen, sol_logger: Logger, gen_logger: Logger):
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
    solver_training = True
    logger = sol_logger
    for episode in range(opt.episodes):
        current_state = env.reset()
        current_state_input, last_state = preprocess(None, current_state, cnn=True)
        logger.episode_start(episode)
        done = False
        total_reward = 0.0

        while not done:
            action, action_prob, state_value = solver.take_action(current_state_input)
            next_state, reward, done, info = env.step(action, skip=opt.frame_skipping)

            if solver_training:
                solver.fill_buffer(current_state_input, state_value, action_prob, K.expand_dims(action), reward, int(done))
                total_reward += reward
            elif info["gen"]:
                # The policy network predict just on flags while the value network estimate the expected return with the complete observation.
                # Then on the advantage of returns, the policy is improved.
                state_value = generator.get_state_value(last_state,  info["gen_diff"])
                generator.fill_buffer(last_state, state_value, info["gen_probs"], info["gen_pos"], info["gen_rew"], int(done))
                total_reward += info["gen_rew"]

            # preprocess new state
            current_state_input, last_state = preprocess(current_state_input, next_state, cnn=True)

            not_running = not register_input()
            if done or not_running:
                break

        # add last state value estimate needed for advantage computing and update
        if solver_training:
            _, _, state_value = solver.take_action(current_state_input)
            solver.buffer.state_values.append(state_value)
            # update with data batch collected
            solver.update(logger)
        else:
            state_value = generator.get_state_value(last_state, generator.buffer.difficulties[-1])
            generator.buffer.state_values.append(state_value)
            generator.buffer.partial_states.pop()
            generator.buffer.difficulties.pop()
            generator.buffer.free_positions.pop()
            # update with data batch collected (some data comes also from the environment map generation step)
            generator.update(logger)

        logger.episode_stop(total_reward, env.score_points)

        # optionally switch training
        if opt.generator_train and episode != 0 and (episode % (opt.alternate_training_interval*(1 if solver_training else 3)) == 0):
            # the generator should be trained the double the episodes respect the solver
            solver.freeze_switch()
            generator.freeze_switch()
            solver_training = not solver_training
            logger = sol_logger if solver_training else gen_logger
            logger.total_steps = (gen_logger if solver_training else sol_logger).total_steps
            print("\n\nSwitch training to generator {}".format("solver" if solver_training else "generator"))

        # optionally save models
        if episode != 0 and (episode % opt.agent_save_interval == 0):
            print(f'Saving agent at episode: {episode}.')
            solver.save_model(episode)
            if opt.generator_train: generator.save_model(episode)

        if not_running: break
    env.close()


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

