import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers, initializers


class Actor:
    def __init__(self, time_steps, state_size, action_dim):
        self.time_steps = time_steps
        self.state_size = state_size # observation size
        self.action_dim = action_dim # 행동 사이즈
        if time_steps == 0: # time_steps가 0이면 DNN 사용
            self.model = self.build_model()
        else:
            self.model = self.build_LSTM_model()
        # self.model.summary()

    def build_model(self):
        states = keras.Input(shape=(self.state_size,))
        out = layers.Dense(256, activation="relu",kernel_initializer=initializers.HeNormal())(states)
        out = layers.Dense(128, activation="relu",kernel_initializer=initializers.HeNormal())(out)
        actions = layers.Dense(self.action_dim, activation='softmax')(out)
        model = keras.Model(inputs=states, outputs=actions)
        
        return model
    
    def build_LSTM_model(self):
        states = keras.Input(shape=(self.time_steps, self.state_size))
        out = layers.LSTM(256, return_sequences=True, kernel_initializer=initializers.HeNormal())(states)
        out = layers.LSTM(128, return_sequences=False, kernel_initializer=initializers.HeNormal())(out)
        actions = layers.Dense(self.action_dim, activation='softmax')(out)
        model = keras.Model(inputs=states, outputs=actions)
        return model
    

class Critic:
    def __init__(self, time_steps, state_size):
        self.time_steps = time_steps
        self.state_size = state_size
        if time_steps == 0:
            self.model = self.build_model()
        else:
            self.model = self.build_LSTM_model()

        # self.model.summary()

    def build_model(self):
        states = keras.Input(shape=(self.state_size,))
        out = layers.Dense(256, activation="relu",kernel_initializer=initializers.HeNormal())(states)
        out = layers.Dense(128, activation="relu",kernel_initializer=initializers.HeNormal())(out)
        q_values = layers.Dense(1, activation='linear')(out)
        model = keras.Model(inputs=states, outputs=q_values)
        return model
    
    def build_LSTM_model(self):
        states = keras.Input(shape=(self.time_steps, self.state_size))
        out = layers.LSTM(256, return_sequences=True, kernel_initializer=initializers.HeNormal())(states)
        out = layers.LSTM(128, return_sequences=False, kernel_initializer=initializers.HeNormal())(out)
        q_values = layers.Dense(1, activation='linear')(out)
        model = keras.Model(inputs=states, outputs=q_values)
        return model
    

class A2Cagent:

    def __init__(self, time_steps, state_dim, action_dim, gamma, batch_size):
        # hyperparameters
        self.GAMMA = gamma
        self.BATCH_SIZE = batch_size
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = Actor(time_steps, self.state_dim, self.action_dim)
        self.critic = Critic(time_steps, self.state_dim)

        self.actor_opt = keras.optimizers.Adam(learning_rate=self.ACTOR_LEARNING_RATE)
        self.critic_opt = keras.optimizers.Adam(learning_rate=self.CRITIC_LEARNING_RATE)

        self.save_epi_reward = []

    # action 뱉기
    def get_action(self, state):
        state = np.array([state])
        action_probs = self.actor.model(state)
        probs = action_probs[0].numpy()
        # 이중 안전장치
        probs = np.clip(probs, a_min=0, a_max=None)  # 음수가 있을 경우 0으로 설정
        probs /= np.sum(probs)  # 합이 1이 되도록 정규화

        return probs

    def td_target(self, reward, v_value, next_v_value, done):
        if done:
            y_k = tf.constant([[reward]], dtype=tf.float32)
            advantage = y_k - v_value
        else:
            y_k = reward + self.GAMMA * next_v_value
            advantage = y_k - v_value
        return advantage, y_k
    
    def critic_learn(self, states, td_targets):
        with tf.GradientTape() as tape:
            td_hat = self.critic.model(states, training=True)
            loss = tf.reduce_mean(tf.square(td_targets - td_hat))  # q - td_targets 을 통해 loss 계산

        grads = tape.gradient(loss, self.critic.model.trainable_variables)  # loss로 gradient 계산
        # Gradient Cliping
        # grads = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads]
        self.critic_opt.apply_gradients(zip(grads, self.critic.model.trainable_variables))  # critic 조정
    
    def actor_learn(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            probs = self.actor.model(states, training=True)  # states에 대한 action을 뽑아서
            action_probs = tf.reduce_sum(actions * probs, axis=1)
            log_probs = tf.math.log(action_probs + 1e-10)
            loss = -tf.reduce_mean(log_probs * advantages) # critic_q에 대한 loss계산

        grads = tape.gradient(loss, self.actor.model.trainable_variables)
        # Gradient Cliping
        grads = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads]
        self.actor_opt.apply_gradients(zip(grads, self.actor.model.trainable_variables))

