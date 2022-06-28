import pickle
import tensorflow as tf
from os import remove
from os.path import basename, dirname, join
from random import sample
from typing import Union
from zipfile import ZipFile

import numpy as np
from keras import Model
from keras.callbacks import History
from keras.models import load_model
from keras.models import clone_model
from keras.utils.np_utils import to_categorical
from keras import backend as K
from core.model import AC
from core.policy import EGreedyPolicy


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.state_values = []
        self.action_probs = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []


class GAE_PPO(object):
    def __init__(self, opt, model, frozen: bool = False):
        self.opt = opt
        self.adv_gamma = opt.adv_gamma  # 0.99
        self.adv_lambda = opt.adv_lambda  # 0.95
        self.K_epochs = opt.K_epochs  # 10
        self.batch_size = opt.batch_size  # 128

        self.buffer = RolloutBuffer()
        self.is_frozen = frozen

        self.actor_critic = model

    def fill_buffer(self, state, state_value, action_prob, action, reward, termination_status):
        self.buffer.states.append(state)
        self.buffer.state_values.append(state_value)
        self.buffer.action_probs.append(action_prob)
        self.buffer.actions.append(action)
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(termination_status)

    def _formalize_buffer(self):
        # tf.reshape(action_prob[0, action], (1, 1))
        self.buffer.states = tf.stack(self.buffer.states)
        self.buffer.state_values = tf.squeeze(tf.stack(self.buffer.state_values),1)
        self.buffer.action_probs = K.expand_dims(self.buffer.action_probs, 1)
        self.buffer.actions = K.expand_dims(tf.convert_to_tensor(self.buffer.actions, dtype=tf.int32), 1)
        self.buffer.rewards = K.expand_dims(tf.convert_to_tensor(self.buffer.rewards, dtype=tf.float32), 1)
        self.buffer.is_terminals = K.expand_dims(tf.convert_to_tensor(self.buffer.is_terminals, dtype=tf.float32), 1)

    def take_action(self, state):
        if (len(state.shape) == 3 and self.opt.cnn>0) or (len(state.shape) == 1 and (not self.opt.cnn>0)):
            state = K.expand_dims(state, 0)
        # ac, b = self.actor_critic.predict(state)  # action_prob, state_value
        # a = np.zeros((1, 5))
        # a[:, np.argmax(ac)] = 1
        # return a, b
        return self.actor_critic.predict(state)

    def update(self, logger):
        """Update the model parameters fitting for `K_epochs` on the date in the `buffer` """
        def get_advantages(mc_rollout_method=False):
            """ GAE PPO advantage given by delta factor i.e. the TD residual of V with discount gamma.
             It can be considered as an estimate of the advantage of an action 'a' at time t.
            """
            returns = []
            discounted_reward = 0

            if mc_rollout_method:
                # Monte Carlo estimate of returns
                for total_reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
                    if is_terminal:
                        discounted_reward = 0
                    discounted_reward = total_reward + (self.adv_gamma * discounted_reward)
                    returns.append(discounted_reward)
            else:
                # PPO general estimate of returns
                for i in reversed(range(len(self.buffer.rewards))):
                    delta = self.buffer.rewards[i] + self.adv_gamma * self.buffer.state_values[i + 1] * (1-self.buffer.is_terminals[i]) - self.buffer.state_values[i]
                    discounted_reward = delta + self.adv_gamma * self.adv_lambda * (1-self.buffer.is_terminals[i]) * discounted_reward
                    returns.append(discounted_reward + self.buffer.state_values[i])
            returns.reverse()
            returns = tf.stack(returns)
            advantages = returns - self.buffer.state_values[:-1]
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

            return returns, advantages

        self._formalize_buffer()
        expected_cum_returns, advantages = get_advantages()

        evenly_distributed_log = list(map(lambda x : len(x), np.array_split(np.array((range(len(self.buffer.rewards)))), self.K_epochs)))
        batch_indices = np.arange(len(self.buffer.rewards))
        for i in range(self.K_epochs):
            np.random.shuffle(batch_indices)
            losses = []
            for lower_end in [a for a in range(len(self.buffer.rewards))][::self.batch_size]:
                higher_end = lower_end+self.batch_size
                minibatch_indices = batch_indices[lower_end:higher_end]
                actor_loss, critic_loss = self.actor_critic.fit(
                    tf.gather(self.buffer.states, minibatch_indices),
                    tf.gather(self.buffer.action_probs, minibatch_indices),
                    tf.gather(self.buffer.actions, minibatch_indices),
                    tf.gather(advantages, minibatch_indices),
                    tf.gather(self.buffer.state_values[:-1], minibatch_indices),
                    tf.gather(expected_cum_returns, minibatch_indices))
                losses.append({"actor_loss": actor_loss, "critic_loss": critic_loss})
            logger.step(evenly_distributed_log[i], losses, self.actor_critic.a_optimizer.lr.numpy())

        # Reinitialize buffer
        self.buffer = RolloutBuffer()

        return losses

    def freeze_switch(self):
        self.is_frozen = not self.is_frozen

    def save_model(self, model_episode: int):
        self.actor_critic.save(model_episode)
