import numpy as np
import os
from tensorflow import keras
from collections import deque
import zmq
import random
import logging

class ActorCriticAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.actor_model = self._build_actor_model()
        self.critic_model = self._build_critic_model()
        logging.basicConfig(level=logging.INFO)

    def _build_actor_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def _build_critic_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        policy = self.actor_model.predict(state)[0]
        return np.random.choice(self.action_size, p=policy)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * self.critic_model.predict(next_state)[0]
            target_f = self.critic_model.predict(state)
            target_f[0] = target
            self.critic_model.fit(state, target_f, epochs=1, verbose=0)
            
            advantages = np.zeros((1, self.action_size))
            advantages[0][action] = target - self.critic_model.predict(state)[0]
            self.actor_model.fit(state, advantages, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, actor_name, critic_name):
        self.actor_model.load_weights(actor_name)
        self.critic_model.load_weights(critic_name)

    def save(self, actor_name, critic_name):
        self.actor_model.save_weights(actor_name)
        self.critic_model.save_weights(critic_name)

if __name__ == "__main__":
    state_size = 4
    action_size = 2
    agent = ActorCriticAgent(state_size, action_size)
    
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    try:
        socket.bind(f"tcp://*:{os.environ['PORT']}")

        while True:
            message = socket.recv_json()
            state = np.array(message['state']).reshape([1, state_size])
            action = agent.act(state)
            socket.send_string(str(action))
            next_state = np.array(message['next_state']).reshape([1, state_size])
            reward = message['reward']
            done = message['done']
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > 32:
                agent.replay(32)
            logging.info(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        socket.close()
        context.term()