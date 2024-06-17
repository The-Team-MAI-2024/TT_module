import numpy as np
import os
from tensorflow import keras
from collections import deque
import zmq
import random
import logging

class RLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        logging.basicConfig(level=logging.INFO)

    def _build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    state_size = 4
    action_size = 2
    agent = RLAgent(state_size, action_size)
    
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
                agent.update_target_model()
            logging.info(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        socket.close()
        context.term()