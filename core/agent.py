from argparse import Namespace

import tensorflow as tf

import numpy as np
from keras import backend as K
from core.model import AC
from env import GRID_SIZE, MAX_STEPS
from utils.logger import Logger


class RolloutBuffer:
    def __init__(self):
        self.partial_states = []
        self.states = []
        self.state_values = []
        self.action_probs = []
        self.actions = []
        self.difficulties = []
        self.free_positions = []
        self.rewards = []
        self.is_terminals = []


class PPO(object):
    def __init__(self, opt: Namespace, model: AC, frozen: bool = False):
        self.opt = opt
        self.adv_gamma = opt.adv_gamma  # 0.99
        self.adv_lambda = opt.adv_lambda  # 0.95
        self.K_epochs = opt.K_epochs  # 10
        self.batch_size = opt.batch_size  # 64

        self.buffer = RolloutBuffer()
        self.is_frozen = frozen
        self.is_evaluating = False

        self.actor_critic = model

    def fill_buffer(self, state, state_value, action_prob, action, reward, termination_status):
        """Data collected from a step interaction with the environment."""
        self.buffer.states.append(state)
        self.buffer.state_values.append(state_value)
        self.buffer.action_probs.append(action_prob)
        self.buffer.actions.append(action)
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(termination_status)

    def _formalize_buffer(self):
        """Set the shape of the objects in the buffer to be used for training."""
        self.buffer.partial_states = tf.stack(self.buffer.partial_states)  # (0,) or (s,224,224,1)
        self.buffer.states = tf.stack(self.buffer.states)  # (s,224,224,4) or (s,224,224,1)
        self.buffer.state_values = tf.stack(self.buffer.state_values)  # (s+1,1)
        self.buffer.action_probs = K.stack(self.buffer.action_probs)  # (s,1) or (s,n)
        self.buffer.actions = K.stack(self.buffer.actions)  # (s,1) or (s,n)
        self.buffer.difficulties = tf.stack(self.buffer.difficulties)  # (0,) or (s,1)
        self.buffer.free_positions = tf.stack(self.buffer.free_positions)  # (0,) or (s,grid*grid)
        self.buffer.rewards = K.expand_dims(tf.convert_to_tensor(self.buffer.rewards, dtype=tf.float32), 1)  # (s,1)
        self.buffer.is_terminals = K.expand_dims(tf.convert_to_tensor(self.buffer.is_terminals, dtype=tf.float32), 1)  # (s,1)

    def _get_advantages(self, mc_rollout_method=False):
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
                delta = self.buffer.rewards[i] + self.adv_gamma * self.buffer.state_values[i + 1] * (
                            1 - self.buffer.is_terminals[i]) - self.buffer.state_values[i]
                discounted_reward = delta + self.adv_gamma * self.adv_lambda * (
                            1 - self.buffer.is_terminals[i]) * discounted_reward
                returns.append(discounted_reward + self.buffer.state_values[i])
        returns.reverse()
        returns = tf.stack(returns)
        advantages = returns - self.buffer.state_values[:-1]
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

        return returns, advantages

    def update(self, logger):
        pass

    def freeze_switch(self):
        """Switch the internal behaviour to freeze training"""
        self.is_frozen = not self.is_frozen

    def eval_switch(self):
        """Switch the internal behaviour to evaluation"""
        self.is_evaluating = not self.is_evaluating

    def save_model(self, model_episode: int):
        """Save the model"""
        self.actor_critic.save(model_episode)


class PPO_sol(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, logger):
        """Update the model parameters fitting for `K_epochs` on the date in the `buffer`.
        :param logger: logger object to record the training.
        :return: losses values.
        """
        self._formalize_buffer()
        expected_cum_returns, advantages = self._get_advantages()

        evenly_distributed_log = list(
            map(lambda x: len(x), np.array_split(np.array((range(len(self.buffer.rewards)))), self.K_epochs)))
        batch_indices = np.arange(len(self.buffer.rewards))
        for i in range(self.K_epochs):
            np.random.shuffle(batch_indices)
            losses = []
            for lower_end in [a for a in range(MAX_STEPS)][::self.batch_size]:
                higher_end = lower_end + self.batch_size
                minibatch_indices = batch_indices[lower_end:higher_end]
                actor_loss, critic_loss = self.actor_critic.fit(
                    actor_input=(tf.gather(self.buffer.states, minibatch_indices),),
                    critic_input=tf.gather(self.buffer.states, minibatch_indices),
                    actions_prob=tf.gather(self.buffer.action_probs, minibatch_indices),
                    actions=tf.gather(self.buffer.actions, minibatch_indices),
                    advantages=tf.gather(advantages, minibatch_indices),
                    state_values=tf.gather(self.buffer.state_values[:-1], minibatch_indices),
                    returns=tf.gather(expected_cum_returns, minibatch_indices))
                losses.append({"actor_loss": actor_loss, "critic_loss": critic_loss})
            logger.step(evenly_distributed_log[i], losses, self.actor_critic.a_optimizer.lr.numpy())

        # Reinitialize buffer
        self.buffer = RolloutBuffer()

        return losses

    def take_action(self, state):
        """
        Call the underlying model to get the prediction of the action with probabilities and state value.

        NOTE:
        - If `frozen` returns only the actor result (probability and sampled action).
        - (NOPE) If `evaluate` returns only the action took as argmax of the probability.
        :param state: state (224,224,4)
        :return: predicted state value (1,)
        """
        state = tf.expand_dims(state, 0)  # add batch dimension 1,224,224,4
        state_value = None
        if not self.is_frozen and not self.is_evaluating:
            state_value = self.actor_critic.predict_critic(state)[0]

        action_probs = self.actor_critic.predict_actor(state)
        # if self.is_frozen:  # TODO make it more deterministic?
        #     action = np.argmax(action_probs)
        #     action_prob = None
        # else:
        action = np.random.choice(self.actor_critic.action_space_size, p=action_probs[0, :].numpy())
        action_prob = action_probs[:1, action]

        return action, action_prob, state_value  # (1)  (1,)  (1,)


class PPO_gen(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_generations = 5

    def partial_fill_buffer(self, partial_state, difficulty, free_map):
        """Data used to make positions predictions into the environment context."""
        self.buffer.partial_states.append(partial_state)
        self.buffer.difficulties.append(difficulty)
        self.buffer.free_positions.append(free_map)

    def update(self, logger: Logger):
        """Update the model parameters fitting for `K_epochs` on the date in the `buffer`.
        :param logger: logger object to record the training.
        :return: losses values.
        """
        self._formalize_buffer()
        expected_cum_returns, advantages = self._get_advantages()

        evenly_distributed_log = list(
            map(lambda x: len(x), np.array_split(np.array((range(len(self.buffer.rewards)))), self.K_epochs)))
        batch_indices = np.arange(len(self.buffer.rewards))
        for i in range(self.K_epochs):
            np.random.shuffle(batch_indices)
            losses = []
            for lower_end in [a for a in range(MAX_STEPS)][::self.batch_size]:
                higher_end = lower_end + self.batch_size
                minibatch_indices = batch_indices[lower_end:higher_end]
                actor_loss, critic_loss = self.actor_critic.fit(
                    actor_input=[tf.gather(self.buffer.partial_states, minibatch_indices), tf.gather(self.buffer.difficulties, minibatch_indices), tf.gather(self.buffer.free_positions, minibatch_indices)],
                    critic_input=[tf.gather(self.buffer.states, minibatch_indices), tf.gather(self.buffer.difficulties, minibatch_indices)],
                    actions_prob=tf.gather(self.buffer.action_probs, minibatch_indices),
                    actions=tf.gather(self.buffer.actions, minibatch_indices),
                    advantages=tf.gather(advantages, minibatch_indices),
                    state_values=tf.gather(self.buffer.state_values[:-1], minibatch_indices),
                    returns=tf.gather(expected_cum_returns, minibatch_indices))
                losses.append({"actor_loss": actor_loss, "critic_loss": critic_loss})
            logger.step(evenly_distributed_log[i], losses, self.actor_critic.a_optimizer.lr.numpy())

        # Reinitialize buffer
        self.buffer = RolloutBuffer()

        return losses

    def get_policy_probs(self, *inputs):
        """
        Call the underlying model to get `n_generations` predicted positions with relative probabilities.
        :param inputs: state (224,224,1), difficulty (1) free_map (100).
        :return: position indexes of sampled positions into probability (n,),
                 probabilities of sampled positions (n,),
                 position indexes formatted as coordinates (n,2).
        """
        inputs = [tf.expand_dims(i, 0) for i in inputs]  # add batch dimension (1,224,224,1)  (1,1)  (1,100)
        positions_probs = self.actor_critic.predict_actor(inputs)
        positions = tf.squeeze(tf.random.categorical(tf.math.log(positions_probs), self.n_generations))
        positions_prob = tf.gather(tf.squeeze(positions_probs), positions)
        map_positions = tf.stack([positions // GRID_SIZE, positions % GRID_SIZE], 1)
        # positions = tf.where(tf.reshape(positions_probs, (GRID_SIZE, GRID_SIZE)) > 0.02)  # TODO use a threshold?
        # positions_prob = tf.expand_dims(positions_probs[positions_probs > 0.02], 0)
        return positions, positions_prob, map_positions  # (n,)  (n,)  (n,2)

    def get_state_value(self, *inputs):
        """
        Call the underlying model to get the prediction of the state value.
        :param inputs: state (1,224,224), difficulty (1).
        :return: predicted state value (1,).
        """
        inputs = [tf.expand_dims(i, 0) for i in inputs]  # add batch dimension (1,224,224,1)  (1,1)
        state_value = self.actor_critic.predict_critic(inputs)[0]
        return state_value  # (1,)
