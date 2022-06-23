from typing import Union
from keras import backend as K
import numpy as np
from keras import Input, Model, Sequential
from keras.callbacks import TensorBoard
from keras.layers import Lambda, Flatten, Dense, Multiply, ConvLSTM2D, BatchNormalization, Conv2D
from keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adadelta, Adamax
import tensorflow as tf
from env import STATE_W, STATE_H, LAYERS_ENTITIES, CNN_STATE_W, CNN_STATE_H
from keras.applications.mobilenet_v2 import MobileNetV2


def create_actor_cnn(shape: tuple, action_size: int, optimizer, training=False) -> Model:
    states = Input(shape, name='input')
    oldpolicy_probs = Input(action_size)
    advantages = Input(1)
    values = Input(1)
    returns = Input(1)
    conv1 = Conv2D(32,(8,8),strides=(4,4), activation='relu')(states)
    conv2 = Conv2D(64,(4,4),strides=(2,2), activation='relu')(conv1)
    conv3 = Conv2D(64,(3,3),strides=(1,1), activation='relu')(conv2)
    features = Flatten(name='flatten')(conv3)
    dense = Dense(512, activation='relu', name='fc1')(features)
    output = Dense(action_size, activation='softmax', name='predictions')(dense)
    loss_layer = tf.keras.layers.Lambda(ppo_loss3)([oldpolicy_probs, advantages, values, returns, output])
    # Create the model
    if training:
        model = Model(inputs=[states, oldpolicy_probs, advantages, values, returns], outputs=loss_layer)
    else:
        model = Model(inputs=[states], outputs=[output])

    model.compile(optimizer=optimizer, loss=[dummy_loss])

    return model


def create_critic_cnn(shape: int, optimizer) -> Model:
    states = Input(shape, name='input')
    # Input passed to dense layer2.
    conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(states)
    conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
    conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
    features = Flatten(name='flatten')(conv3)
    dense = Dense(512, activation='relu', name='fc1')(features)
    output = Dense(1, activation='tanh', name='output')(dense)

    def custom_layer(x):
        return tf.math.multiply(x, 1000)
    output_scaled = Lambda(custom_layer, name="lambda_layer")(output)
    # Create the model
    model = Model(inputs=states, outputs=output_scaled)

    model.compile(optimizer=optimizer, loss='mse')

    return model


def create_actor_image(shape: tuple, action_size: int) -> Model:
    states = Input(shape)

    feature_extractor = MobileNetV2(input_shape=shape, include_top=False, weights='imagenet')
    # Freeze layers
    for layer in feature_extractor.layers:
        layer.trainable = False
    # Classification block
    features = Flatten(name='flatten')(feature_extractor(states))
    dense = Dense(1024, activation='relu', name='fc1')(features)
    output = Dense(action_size, activation='softmax', name='predictions')(dense)
    # Create the model
    model = Model(inputs=[states], outputs=[output])
    model.compile()

    return model


def create_critic_image(shape: tuple) -> Model:
    states = Input(shape)

    feature_extractor = MobileNetV2(input_shape=shape, include_top=False, weights='imagenet')
    # Freeze layers
    for layer in feature_extractor.layers:
        layer.trainable = False
    # Classification block
    features = Flatten(name='flatten')(feature_extractor(states))
    dense = Dense(1024, activation='relu', name='fc1')(features)
    output = Dense(1, activation='tanh')(dense)

    def custom_layer(x):
        return tf.math.multiply(x, 1000)
    output_scaled = Lambda(custom_layer, name="lambda_layer")(output)
    # Create the model
    model = Model(inputs=[states], outputs=[output_scaled])
    model.compile()

    return model


def create_actor(shape: int, action_size: int, optimizer, training=False) -> Model:
    states = Input(shape, name='input')
    oldpolicy_probs = Input(action_size)
    advantages = Input(1)
    values = Input(1)
    returns = Input(1)
    # Input passed to dense layer2.
    dense = Dense(512, activation='relu', name='dense1')(states)
    dense = Dense(512, activation='relu', name='dense2')(dense)
    output = Dense(action_size, activation='softmax', name='output')(dense)
    loss_layer = tf.keras.layers.Lambda(ppo_loss3)([oldpolicy_probs, advantages, values, returns, output])
    # Create the model
    if training:
        model = Model(inputs=[states, oldpolicy_probs, advantages, values, returns], outputs=loss_layer)
    else:
        model = Model(inputs=[states], outputs=[output])

    model.compile(optimizer=optimizer, loss=[dummy_loss])

    return model


def create_critic(shape: int, optimizer) -> Model:
    states = Input(shape, name='input')
    # Input passed to dense layer2.
    dense = Dense(512, activation='relu', name='dense1')(states)
    dense = Dense(512, activation='relu', name='dense2')(dense)
    output = Dense(1, activation='tanh', name='output')(dense)

    def custom_layer(x):
        return tf.math.multiply(x, 1000)
    output_scaled = Lambda(custom_layer, name="lambda_layer")(output)
    # Create the model
    model = Model(inputs=states, outputs=output_scaled)

    model.compile(optimizer=optimizer, loss='mse')

    return model


class AC:
    def __init__(self, opt, action_space_size):
        self.action_space_size = action_space_size

        self.logger = TensorBoard()
        self.a_optimizer = initialize_optimizer(opt.optimizer, opt.learning_rate, opt.lr_beta1, opt.lr_beta2,
                                                opt.learning_rate_decay, opt.lr_rho, opt.lr_fuzz, opt.lr_momentum)
        self.c_optimizer = initialize_optimizer(opt.optimizer, opt.learning_rate, opt.lr_beta1, opt.lr_beta2,
                                                opt.learning_rate_decay, opt.lr_rho, opt.lr_fuzz, opt.lr_momentum)

        self.ph_n = np.zeros((1, self.action_space_size))
        self.ph_1 = np.zeros((1, 1))

        if opt.cnn:
            input_dim = (CNN_STATE_H, CNN_STATE_W, 3)
            self.actor_t = create_actor_cnn(input_dim, 5, self.a_optimizer, training=True)
            self.actor_p = create_actor_cnn(input_dim, 5, self.a_optimizer, training=False)
            self.critic = create_critic_cnn(input_dim, self.c_optimizer)
        else:
            input_dim = STATE_W*STATE_H*LAYERS_ENTITIES
            self.actor_t = create_actor(input_dim, 5, self.a_optimizer, training=True)
            self.actor_p = create_actor(input_dim, 5, self.a_optimizer, training=False)
            self.critic = create_critic(input_dim, self.c_optimizer)

    def predict(self, state):

        action_dist = self.actor_p.predict([state], verbose=False)
        state_value = self.critic.predict([state], verbose=False)

        return action_dist, state_value

    def fit(self, states, actions_probs, advantages, state_values, returns, k, b):
        # with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
        #     mse = tf.keras.losses.MeanSquaredError()
        #     pred_values_estim = self.critic([states], training=True)
        #     critic_loss = clip_gamma * mse(returns, pred_values_estim)
        #     pred_probs = self.actor([states], training=True)
        #     actor_loss = ppo_loss(actions_probs, pred_probs, advantages, critic_loss.numpy())
        # grads1 = tape1.gradient(actor_loss, self.actor.trainable_variables)
        # grads2 = tape2.gradient(critic_loss, self.critic.trainable_variables)
        # self.a_optimizer.apply_gradients(zip(grads1, self.actor.trainable_variables))
        # self.c_optimizer.apply_gradients(zip(grads2, self.critic.trainable_variables))

        critic_loss = self.critic.fit(
            [states],
            [np.reshape(returns, newshape=(-1, 1))],
            epochs=k, shuffle=True, verbose=True, batch_size=b,
            callbacks=[self.logger])

        actor_loss = self.actor_t.fit(
            [states, actions_probs, advantages, state_values, returns],
            [actions_probs],  # dummy
            epochs=k, shuffle=True, verbose=True, batch_size=b,
            callbacks=[self.logger])
        # Weights transfer
        self.actor_p.set_weights(self.actor_t.get_weights())

        return actor_loss, critic_loss

# def atari_skiing_model(shape: tuple, action_size: int, optimizer: Optimizer) -> Model:
#     """
#     Defines a Keras Model designed for the atari skiing game.
# 
#     :param shape: the input shape.
#     :param action_size: the number of available actions.
#     :param optimizer: an optimizer to be used for model compilation.
#     :return: the Keras Model.
#     """
#     # Create the input layers.
#     inputs = Input(shape, name='input')
#     actions_input = Input((action_size,), name='input_mask')
#     # Create a normalization layer.
#     normalized = Lambda(lambda x: x / 255.0, name='normalisation')(inputs)
# 
#     # Create CNN-LSTM layers.
#     conv_lstm2d_1 = ConvLSTM2D(16, (8, 8), strides=(4, 4), activation='relu', return_sequences=True,
#                                name='conv_lstm_2D_1')(normalized)
#     batch_norm = BatchNormalization(name='batch_norm1')(conv_lstm2d_1)
#     conv_lstm2d_2 = ConvLSTM2D(32, (4, 4), strides=(2, 2), activation='relu', name='conv_lstm_2D_2')(batch_norm)
#     batch_norm2 = BatchNormalization(name='batch_norm2')(conv_lstm2d_2)
# 
#     # Flatten the output and pass it to a dense layer.
#     flattened = Flatten(name='flatten')(batch_norm2)
#     dense = Dense(256, activation='relu', name='dense1')(flattened)
# 
#     # Create and filter the output, multiplying it with the actions input mask, in order to get the QTable.
#     output = Dense(action_size, name='dense2')(dense)
#     filtered_output = Multiply(name='filtered_output')([output, actions_input])
# 
#     # Create the model.
#     model = Model(inputs=[inputs, actions_input], outputs=filtered_output)
#     # Compile the model.
#     model.compile(optimizer, loss=huber_loss)
# 
#     return model


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
    if optimizer_name == 'adam':
        return Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2, decay=lr_decay)
    elif optimizer_name == 'rmsprop':
        return RMSprop(learning_rate=learning_rate, rho=rho, epsilon=fuzz)
    elif optimizer_name == 'sgd':
        return SGD(learning_rate=learning_rate, momentum=momentum, decay=lr_decay)
    elif optimizer_name == 'adagrad':
        return Adagrad(learning_rate=learning_rate, decay=lr_decay)
    elif optimizer_name == 'adadelta':
        return Adadelta(learning_rate=learning_rate, rho=rho, decay=lr_decay)
    elif optimizer_name == 'adamax':
        return Adamax(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2, decay=lr_decay)
    else:
        raise ValueError('An unexpected optimizer name has been encountered.')


# def huber_loss(y_true, y_pred):
#     """
#     Define the huber loss.
# 
#     :param y_true: the true value.
#     :param y_pred: the predicted value.
#     :return: a tensor with the result.
#     """
#     # Calculate the error.
#     error = abs(y_true - y_pred)
# 
#     # Calculate MSE.
#     quadratic_term = error * error / 2
#     # Calculate MAE.
#     linear_term = error - 1 / 2
# 
#     # Use mae if |error| > 1.
#     use_linear_term = (error > 1.0)
#     # Cast the boolean to float, in order to be compatible with Keras.
#     use_linear_term = cast(use_linear_term, 'float32')
# 
#     # Return MAE or MSE based on the flag.
#     return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term


def ppo_loss(oldpolicy_probs, newpolicy_predicted_probs, advantages, critic_loss):
    """
    The actor loss is the minimum between the ratio of policies x advantage ant the clipped ratio x advantage.
    This disincentivize the new policy to get far from the old policy. Note that the ratio > 1 if the action will
    be selected more in the new policy, and ratio < 1 if the action will be selected less in the new policy.

    Combining the critic network loss help (as discount factor) making updates for both networks with the same order
     of magnitude

    The use of an entropy term encourages our actor model to explore different policies
    """
    clip_eps = 0.2  # from PPO paper
    entropy_beta = 0.01

    newpolicy_probs = newpolicy_predicted_probs
    ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10),)
    p1 = ratio * advantages
    p2 = K.clip(ratio, min_value=1 - clip_eps, max_value=1 + clip_eps) * advantages
    actor_loss = -K.mean(K.minimum(p1, p2))
    entropy_term = K.mean(-(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
    total_loss = actor_loss + critic_loss - (entropy_beta * entropy_term)
    return total_loss

def ppo_loss3(tensor):
    clip_eps = 0.2  # from PPO paper
    critic_gamma = 0.5  # for critic from PPO paper 
    entropy_beta = 0.01

    oldpolicy_probs, advantages, values, returns, y_pred = tensor[0], tensor[1], tensor[2], tensor[3], tensor[4]
    newpolicy_probs = y_pred
    ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
    p1 = ratio * advantages
    p2 = K.clip(ratio, min_value=1 - clip_eps, max_value=1 + clip_eps) * advantages
    actor_loss = -K.mean(K.minimum(p1, p2))
    critic_loss = K.mean(K.square(returns - values))
    entropy_term = K.mean(-(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
    total_loss = (critic_gamma * critic_loss) + actor_loss - (entropy_beta * entropy_term)
    return total_loss

def dummy_loss(y_true, y_pred):
    return y_pred