import pickle
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


class ExperienceReplayMemory(object):
    """ Implements a Ring Buffer with an extra function which randomly samples elements from it,
    in order to be used as an Experience Replay Memory for the agent. """

    def __init__(self, size: int):
        # Check size value.
        if size < 1:
            raise ValueError('Memory size must be a positive integer. Got {} instead.'.format(size))

        # Initialize array.
        # Allocate one extra element, so that self.start == self.end always means the buffer is EMPTY,
        # whereas if exactly the right number of elements is allocated,
        # it also means the buffer is full. This greatly simplifies the rest of the code.
        self.data = [None] * (size + 1)
        # Initialize start pointer.
        self.start = 0
        # Initialize end pointer.
        self.end = 0

    def append(self, element) -> None:
        """
        Appends an element to the memory.

        :param element: the element to append.
        """
        # Add the element to the end of the memory.
        self.data[self.end] = element
        # Increment the end pointer.
        self.end = (self.end + 1) % len(self.data)

        # Remove the first element by incrementing start pointer, if the memory size has been reached.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def randomly_sample(self, num_items: int) -> list:
        """
        Samples a number of items from the memory randomly.

        :param num_items: the number of the random items to be sampled.
        :return: the items.
        """
        # Sample a random number of memory indexes, which result in non empty contents and return the contents.
        indexes = sample(range(len(self)), num_items)
        return [self[idx] for idx in indexes]

    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class DQN(object):
    def __init__(self, model: Model, target_model_change: int, gamma: float, batch_size: int,
                 observation_space_shape: tuple, action_size: int, policy: EGreedyPolicy, target_model: Model = None,
                 memory_size: int = None, memory: ExperienceReplayMemory = None):
        self.model = model
        self.target_model_change = target_model_change
        self.memory = ExperienceReplayMemory(memory_size) if memory is None else memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.observation_space_shape = observation_space_shape
        self.action_size = action_size
        self.policy = policy
        self.target_model = self._create_target_model() if target_model is None else target_model
        self.steps_from_update = 0

    def _create_target_model(self) -> Model:
        """
        Creates the target model, by copying the model.

        :return: the target model.
        """
        target_model = clone_model(self.model)
        target_model.build(self.observation_space_shape)
        target_model.compile(optimizer=self.model.optimizer, loss=self.model.loss)
        target_model.set_weights(self.model.get_weights())

        return target_model

    def _get_mini_batch(self) -> [np.ndarray]:
        """
        Samples a random mini batch from the replay memory.

        :return: the current state batch, the actions batch, the rewards batch and the next state batch
        """
        # Randomly sample a mini batch.
        mini_batch = self.memory.randomly_sample(self.batch_size)

        # Initialize arrays.
        current_state_batch, next_state_batch, actions, rewards = \
            np.empty(((self.batch_size,) + self.observation_space_shape)), \
            np.empty(((self.batch_size,) + self.observation_space_shape)), \
            np.empty(self.batch_size), \
            np.empty(self.batch_size)

        # Get values from the mini batch.
        for idx, val in enumerate(mini_batch):
            current_state_batch[idx] = val[0]
            actions[idx] = val[1]
            rewards[idx] = val[2]
            next_state_batch[idx] = val[3]

        return current_state_batch, actions, rewards, next_state_batch

    def take_action(self, current_state: np.ndarray, episode: int) -> int:
        """
        Takes an action based on the policy.

        :param current_state: the state for which the action will be taken.
        :param episode: the current episode.
        :return: the action number.
        """
        return self.policy.take_action(self.model, current_state, episode)

    def append_to_memory(self, current_state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        """
        Adds values to the agent's memory.

        :param current_state: the state to add.
        :param action: the action to add.
        :param reward: the total_reward to add.
        :param next_state: the next state to add.
        """
        self.memory.append((current_state, action, reward, next_state))

    def update_target_model(self) -> None:
        """ Updates the target model. """
        self.target_model.set_weights(self.model.get_weights())
        self.steps_from_update = 0

    def fit(self) -> Union[History, None]:
        """
        Fits the agent.

        :return: the fit history.
        """
        # Fit only if the agent is not observing.
        if not self.policy.observing:
            # Increase the steps from update indicator.
            self.steps_from_update += 1

            # Get the mini batches.
            current_state_batch, actions, rewards, next_state_batch = self._get_mini_batch()

            # Create the actions mask.
            actions_mask = np.ones((self.batch_size, self.action_size))
            # Predict the next QValues.
            next_q_values = self.target_model.predict([next_state_batch, actions_mask])
            # Initialize the target QValues for the mini batch.
            target_q_values = np.empty((self.batch_size,))

            for i in range(self.batch_size):
                # Update rewards, using the Deep Q Learning rule.
                target_q_values[i] = rewards[i] + self.gamma * np.amax(next_q_values[i])

            # One hot encode the actions.
            one_hot_actions = to_categorical(actions, self.action_size)
            # One hot encode the target QValues.
            one_hot_target_q_values = one_hot_actions * np.expand_dims(target_q_values, 1)

            # Fit the model to the batches.
            history = self.model.fit([current_state_batch, one_hot_actions], one_hot_target_q_values, epochs=1, batch_size=self.batch_size, verbose=0)

            # Update the target model if necessary.
            if self.steps_from_update == self.target_model_change or self.target_model_change < 1:
                print('Updating target model.')
                self.update_target_model()
                print('Target model has been successfully updated.')

            return history

    def save_agent(self, filename_prefix: str) -> str:
        """
        Saves the agent.

        :param filename_prefix: the agent's filename prefix.
        :return: the filename.
        """
        # Create filenames.
        model_filename = 'model.h5'
        target_model_filename = 'target_model.h5'
        config_filename = 'config.pickle'
        zip_filename = filename_prefix + '.zip'

        # Save models.
        self.model.save(model_filename)
        self.target_model.save(target_model_filename)

        # Create configuration dict.
        config = dict({
            'target_model_change': self.target_model_change,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'observation_space_shape': self.observation_space_shape,
            'action_size': self.action_size,
            'policy': self.policy,
            'memory': self.memory
        })

        # Save configuration.
        with open(config_filename, 'wb') as stream:
            pickle.dump(config, stream, protocol=pickle.HIGHEST_PROTOCOL)

        # Zip models and configuration together.
        with ZipFile(zip_filename, 'w') as model_zip:
            model_zip.write(model_filename, basename(model_filename))
            model_zip.write(target_model_filename, basename(target_model_filename))
            model_zip.write(config_filename, basename(config_filename))

        # Remove files out of the zip.
        remove(model_filename)
        remove(target_model_filename)
        remove(config_filename)

        return zip_filename


def load_dqn_agent(filename: str, custom_objects: dict) -> DQN:
    """
    Loads an agent from a file, using the given parameters.

    :param filename: the agent's filename.
    :param custom_objects: custom_objects for the keras model.
    :return: the DQN agent.
    """
    # Create filenames.
    directory = dirname(filename)
    model_filename = join(directory, 'model.h5')
    target_model_filename = join(directory, 'target_model.h5')
    config_filename = join(directory, 'config.pickle')

    # Read models and memory.
    with ZipFile(filename) as model_zip:
        model_zip.extractall(directory)

    # Load models.
    model = load_model(model_filename, custom_objects=custom_objects)
    target_model = load_model(target_model_filename, custom_objects=custom_objects)

    # Load configuration.
    with open(config_filename, 'rb') as stream:
        config = pickle.load(stream)

    # Remove files out of the zip.
    remove(model_filename)
    remove(target_model_filename)
    remove(config_filename)

    return DQN(model, config['target_model_change'], config['gamma'], config['batch_size'],
               config['observation_space_shape'], config['action_size'], config['policy'], target_model,
               memory=config['memory'])

###---
###---

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.state_values = []
        self.action_probs = []
        self.rewards = []
        self.is_terminals = []

    # def clear(self):
    #     del self.states[:]
    #     del self.state_values[:]
    #     del self.action_probs[:]
    #     del self.rewards[:]
    #     del self.is_terminals[:]


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

    def fill_buffer(self, state, state_value, action_prob, reward, termination_status):
        self.buffer.states.append(state)
        self.buffer.state_values.append(state_value)
        self.buffer.action_probs.append(action_prob)
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(termination_status)

    def _formalize_buffer(self):
        self.buffer.states = np.array(self.buffer.states, dtype=np.float32)
        self.buffer.state_values = np.squeeze(np.array(self.buffer.state_values, dtype=np.float32), 1)
        # self.buffer.state_values = np.array(self.buffer.state_values, dtype=np.float32)
        self.buffer.action_probs = np.squeeze(np.array(self.buffer.action_probs, dtype=np.float32), 1)
        # self.buffer.action_probs = np.array(self.buffer.action_probs, dtype=np.float32)
        self.buffer.rewards = np.expand_dims(np.array(self.buffer.rewards, dtype=np.float32), 1)
        # self.buffer.rewards = np.array(self.buffer.rewards, dtype=np.float32)
        self.buffer.is_terminals = np.array(self.buffer.is_terminals, dtype=np.float32)

    def take_action(self, state):
        if (len(state.shape) == 3 and self.opt.cnn) or (len(state.shape) == 1 and (not self.opt.cnn)):
            state = K.expand_dims(state, 0)
        return self.actor_critic.predict(state)  # action_prob, state_value

    def update(self, logger):
        def get_advantages():
            """ GAE PPO advantage given by delta factor i.e. the TD residual of V with discount gamma.
             It can be considered as an estimate of the advantage of an action 'a' at time t.
            """

            # # Monte Carlo estimate of returns
            # rewards = []
            # discounted_reward = 0
            # for total_reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            #     if is_terminal:
            #         discounted_reward = 0
            #     discounted_reward = total_reward + (self.adv_gamma * discounted_reward)
            #     rewards.insert(0, discounted_reward)
            #     rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-10)

            returns = []
            discounted_reward = 0
            for i in reversed(range(len(self.buffer.rewards))):
                delta = self.buffer.rewards[i] + self.adv_gamma * self.buffer.state_values[i + 1] * (1-self.buffer.is_terminals[i]) - self.buffer.state_values[i]
                discounted_reward = delta + self.adv_gamma * self.adv_lambda * (1-self.buffer.is_terminals[i]) * discounted_reward
                returns.append(discounted_reward + self.buffer.state_values[i])
            returns.reverse()
            returns = np.array(returns, dtype=np.float32)
            advantages = returns - self.buffer.state_values[:-1]

            return returns, (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

        # losses = []
        # self._formalize_buffer()
        # expected_cum_returns, advantages = get_advantages()
        # for s in range(self.K_epochs):
        #     actor_loss, critic_loss = self.actor_critic.fit(
        #         self.buffer.states,
        #         self.buffer.action_probs,
        #         advantages,
        #         self.buffer.state_values[:-1],
        #         expected_cum_returns)
        #     losses.append({"actor_loss": actor_loss, "critic_loss": critic_loss})

        self._formalize_buffer()
        expected_cum_returns, advantages = get_advantages()

        # b_inds = np.arange(len(self.buffer.rewards))
        # for _ in range(self.K_epochs):
        #     np.random.shuffle(b_inds)
        #     losses = []
        #     for lower_end in [a for a in range(len(self.buffer.rewards))][::self.batch_size]:
        #         higher_end = lower_end+self.batch_size
        #         mb_inds = b_inds[lower_end:higher_end]
        #         actor_loss, critic_loss = self.actor_critic.fit(
        #             self.buffer.states[mb_inds],
        #             self.buffer.action_probs[mb_inds],
        #             advantages[mb_inds],
        #             self.buffer.state_values[:-1][mb_inds],
        #             expected_cum_returns[mb_inds]
        #         )
        #         losses.append({"actor_loss": actor_loss, "critic_loss": critic_loss})
        #     logger.step(losses)
        losses = []
        actor_loss, critic_loss = self.actor_critic.fit(
                        self.buffer.states,
                        self.buffer.action_probs,
                        advantages,
                        self.buffer.state_values[:-1],
                        expected_cum_returns, self.K_epochs, self.batch_size)
        # losses.append({"actor_loss": actor_loss, "critic_loss": critic_loss})
        # logger.step(losses)

        # TODO modify the above with this handling
        # TODO change total_reward assigment and reinsert trees
        # TODO leave in execution with trained parameters
        # TODO tomorrow try with a baseline ppo

        # import gym
        # 
        # from stable_baselines3 import PPO
        # from stable_baselines3.common.env_util import make_vec_env
        # 
        # # Parallel environments
        # env = make_vec_env("CartPole-v1", n_envs=4)
        # 
        # model = PPO("MlpPolicy", env, verbose=1)
        # model.learn(total_timesteps=25000)
        # model.save("ppo_cartpole")
        # 
        # del model  # remove to demonstrate saving and loading
        # 
        # model = PPO.load("ppo_cartpole")
        # 
        # obs = env.reset()
        # while True:
        #     action, _states = model.predict(obs)
        #     obs, rewards, dones, info = env.step(action)
        #     env.render()

        # clear buffer
        self.buffer = RolloutBuffer()

        return losses

    def freeze_switch(self):
        self.is_frozen = not self.is_frozen