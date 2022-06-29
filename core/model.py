import glob
import os
from path import Path
from argparse import Namespace
from typing import Union
from keras import backend as K, initializers
import numpy as np
from keras import Input, Model, Sequential
from keras.callbacks import TensorBoard
from keras.layers import Lambda, Flatten, Dense, Concatenate, BatchNormalization, Conv2D, Softmax, Multiply
from keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adadelta, Adamax
import tensorflow as tf
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from keras.saving.save import load_model

from env import CNN_STATE_W, CNN_STATE_H, GRID_SIZE


def create_actor_gen_cnn(shape: tuple, grid_size: int) -> Model:
    # first branch
    states = Input(shape, name='input_state')
    free_pos = Input(grid_size, name='input_positions')
    conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(states)
    conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
    conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
    features = Flatten(name='flatten')(conv3)
    # second branch
    difficulty = Input(shape=(1), name="input_aux")
    merge = Concatenate(axis=1)([features, difficulty])
    # predictor
    dense = Dense(512, activation='relu', name='fc1')(merge)
    output = Dense(grid_size, name='predictions')(dense)
    filtered_output = Multiply(name='filtered_output')([output, free_pos])
    prob_filtered_output = Softmax(name='activation', axis=1)(filtered_output)

    # Create the model
    model = Model(inputs=[states, difficulty, free_pos], outputs=[prob_filtered_output])

    model.compile()

    return model


def create_critic_gen_cnn(shape: tuple) -> Model:
    # first branch
    states = Input(shape, name='input')
    conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(states)
    conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
    conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
    features = Flatten(name='flatten')(conv3)
    # second branch
    difficulty = Input(shape=(1))
    merge = Concatenate(axis=1)([features, difficulty])
    # predictor
    dense = Dense(512, activation='relu', name='fc1')(merge)
    output = Dense(1, name='output', activation=None)(dense)

    # Create the model
    model = Model(inputs=[states, difficulty], outputs=[output])

    model.compile()

    return model


def create_actor_sol_cnn(shape: tuple, action_size: int) -> Model:
    # first branch
    states = Input(shape, name='input')
    conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(states)
    conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
    conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
    features = Flatten(name='flatten')(conv3)
    # predictor
    dense = Dense(512, activation='relu', name='fc1')(features)
    output = Dense(action_size, activation='softmax', name='predictions')(dense)

    # Create the model
    model = Model(inputs=states, outputs=output)

    model.compile()

    return model


def create_critic_sol_cnn(shape: tuple) -> Model:
    # first branch
    states = Input(shape, name='input')
    conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(states)
    conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
    conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
    features = Flatten(name='flatten')(conv3)
    # predictor
    dense = Dense(512, activation='relu', name='fc1')(features)
    output = Dense(1, name='output', activation=None)(dense)

    # Create the model
    model = Model(inputs=states, outputs=output)

    model.compile()

    return model


def create_actor(shape: int, action_size: int) -> Model:
    states = Input(shape, name='input')
    # Input passed to dense layer2.
    dense = Dense(512, activation='relu', name='dense1')(states)
    dense = Dense(512, activation='relu', name='dense2')(dense)
    output = Dense(action_size, activation='softmax', name='output')(dense)
    # Create the model
    model = Model(inputs=states, outputs=output)

    model.compile()

    return model


def create_critic(shape: int) -> Model:
    states = Input(shape, name='input')
    # Input passed to dense layer2.
    dense = Dense(512, activation='relu', name='dense1')(states)
    dense = Dense(512, activation='relu', name='dense2')(dense)
    output = Dense(1, name='output', activation=None)(dense)

    # Create the model
    model = Model(inputs=states, outputs=output)

    model.compile()

    return model


class AC:
    def __init__(self, opt: Namespace, action_space_size: int, load_path: str, load_from_episode: int = None):
        self.action_space_size = action_space_size
        self.save_path = Path(opt.models_path)

        self.a_optimizer = initialize_optimizer(opt.optimizer, opt.lr_a, opt.lr_beta1, opt.lr_beta2,
                                                opt.lr_decay, opt.lr_rho, opt.lr_fuzz, opt.lr_momentum)
        self.c_optimizer = initialize_optimizer(opt.optimizer, opt.lr_c, opt.lr_beta1, opt.lr_beta2,
                                                opt.lr_decay, opt.lr_rho, opt.lr_fuzz, opt.lr_momentum)
        if load_from_episode is not None:
            self._load(load_path, load_from_episode)
        else:
            self.actor = None
            self.critic = None

    def predict_actor(self, inputs):
        prob_dist = self.actor(inputs)
        return prob_dist

    def predict_critic(self, inputs):
        state_value = self.critic(inputs)
        return state_value

    def fit(self, actor_input, critic_input, actions_prob, actions, advantages, state_values, returns):
        critic_gamma = 0.5

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            mse = tf.keras.losses.MeanSquaredError()
            pred_values_estim = self.critic(critic_input, training=True)
            critic_loss = critic_gamma * mse(returns, pred_values_estim)
            pred_probs = self.actor(actor_input, training=True)
            pred_actions_prob = tf.gather(pred_probs, actions, batch_dims=-1)
            total_actor_loss, actor_ratio_loss = ppo_clipped_loss(actions_prob, pred_actions_prob, advantages, state_values, returns)
        grads1 = tape1.gradient(total_actor_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.a_optimizer.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_optimizer.apply_gradients(zip(grads2, self.critic.trainable_variables))

        return actor_ratio_loss, critic_loss

    def save(self, model_episode: int):
        self.actor.save(self.save_path/"{:04}_actor".format(model_episode))
        self.critic.save(self.save_path/"{:04}_critic".format(model_episode))

    def _load(self, path, model_episode: int):
        if model_episode == -1:
            load_filename = '*_actor'
            self.actor = load_model(Path(sorted(glob.glob(os.path.join(path, load_filename)))[-1]))
            load_filename = '*_critic'
            self.critic = load_model(Path(sorted(glob.glob(os.path.join(path, load_filename)))[-1]))
        else:
            load_filename = '{:04}_actor'.format(model_episode)
            self.actor = load_model(Path(path)/load_filename)
            load_filename = '{:04}_critic'.format(model_episode)
            self.critic = load_model(Path(path)/load_filename)
        print("Trained agent loaded")


class AC_gen(AC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = (CNN_STATE_H, CNN_STATE_W, 1)
        self.output_grid = GRID_SIZE * GRID_SIZE
        if self.actor is None:
            self.actor = create_actor_gen_cnn(self.input_dim, self.output_grid)
            self.critic = create_critic_gen_cnn(self.input_dim)


class AC_sol(AC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = (CNN_STATE_H, CNN_STATE_W, 4)
        if self.actor is None:
            self.actor = create_actor_sol_cnn(self.input_dim, self.action_space_size)
            self.critic = create_critic_sol_cnn(self.input_dim)


def initialize_optimizer(optimizer_name: str, learning_rate: float, beta1: float, beta2: float,
                         lr_decay: float, rho: float, fuzz: float, momentum: float) \
        -> Union[Adam, RMSprop, SGD, Adagrad, Adadelta, Adamax]:
    """
    Initializes an optimizer based on the user's choices.

    :param optimizer_name: the optimizer's name.
        Can be one of 'adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adamax'.
    :param learning_rate: the optimizer's learning_rate
    :param beta1: the optimizer's beta1
    :param beta2: the optimizer's beta2
    :param lr_decay: the optimizer's lr_decay
    :param rho: the optimizer's rho
    :param fuzz: the optimizer's fuzz
    :param momentum: the optimizer's momentum
    :return: the optimizer.
    """
    lr_decay_steps = 1
    lr_decay_rate = 1
    if lr_decay:
        schedule = ExponentialDecay(initial_learning_rate=learning_rate, decay_rate=lr_decay_rate, decay_steps=lr_decay_steps)
    else:
        schedule = learning_rate

    if optimizer_name == 'adam':
        return Adam(learning_rate=schedule, beta_1=beta1, beta_2=beta2)
    elif optimizer_name == 'rmsprop':
        return RMSprop(learning_rate=schedule, rho=rho, epsilon=fuzz)
    elif optimizer_name == 'sgd':
        return SGD(learning_rate=schedule, momentum=momentum)
    elif optimizer_name == 'adagrad':
        return Adagrad(learning_rate=schedule)
    elif optimizer_name == 'adadelta':
        return Adadelta(learning_rate=schedule, rho=rho)
    elif optimizer_name == 'adamax':
        return Adamax(learning_rate=schedule, beta_1=beta1, beta_2=beta2)
    else:
        raise ValueError('An unexpected optimizer name has been encountered.')


def ppo_clipped_loss(oldpolicy_probs, newpolicy_predicted_probs, advantages, values, returns):
    """
    The actor loss is the minimum between the ratio of policies x advantage ant the clipped ratio x advantage.
    This disincentivize the new policy to get far from the old policy. Note that the ratio > 1 if the action will
    be selected more in the new policy, and ratio < 1 if the action will be selected less in the new policy.

    Combining the critic network loss help (as discount factor) making updates for both networks with the same order
     of magnitude

    The use of an entropy term encourages our actor model to explore different policies
    """
    clip_eps = 0.2  # from PPO paper
    critic_gamma = 0.5  # for critic from PPO paper 
    entropy_beta = 0.01

    newpolicy_probs = newpolicy_predicted_probs
    ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
    p1 = ratio * advantages
    p2 = K.clip(ratio, min_value=1 - clip_eps, max_value=1 + clip_eps) * advantages
    actor_loss = -K.mean(K.minimum(p1, p2))
    critic_loss = K.mean(K.square(returns - values))
    entropy_term = K.mean(-(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
    total_loss = actor_loss + (critic_gamma * critic_loss) - (entropy_beta * entropy_term)
    return total_loss, actor_loss
