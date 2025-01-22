import tensorflow as tf
import numpy as np
from collections import deque
import random

class QuantumCircuitOptimizer:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Gates Encoder
        gate_input = tf.keras.layers.Input(shape=(None, self.state_size))
        gate_encoder = tf.keras.layers.LSTM(128, return_sequences=True)(gate_input)
        gate_encoder = tf.keras.layers.LSTM(64)(gate_encoder)
        
        # Circuit State Encoder
        state_input = tf.keras.layers.Input(shape=(self.state_size,))
        state_encoder = tf.keras.layers.Dense(64, activation='relu')(state_input)
        
        # Combine encoders
        combined = tf.keras.layers.Concatenate()([gate_encoder, state_encoder])
        
        # Action prediction layers
        dense1 = tf.keras.layers.Dense(128, activation='relu')(combined)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
        output = tf.keras.layers.Dense(self.action_size, activation='linear')(dense2)
        
        model = tf.keras.Model(inputs=[gate_input, state_input], outputs=output)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, gate_state, circuit_state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        gate_state = np.expand_dims(gate_state, axis=0)
        circuit_state = np.expand_dims(circuit_state, axis=0)
        act_values = self.model.predict([gate_state, circuit_state])
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for gate_state, circuit_state, action, reward, next_gate_state, next_circuit_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.target_model.predict([
                        np.expand_dims(next_gate_state, axis=0),
                        np.expand_dims(next_circuit_state, axis=0)
                    ])[0]
                )
            
            target_f = self.model.predict([
                np.expand_dims(gate_state, axis=0),
                np.expand_dims(circuit_state, axis=0)
            ])
            target_f[0][action] = target
            
            self.model.fit(
                [np.expand_dims(gate_state, axis=0), np.expand_dims(circuit_state, axis=0)],
                target_f,
                epochs=1,
                verbose=0
            )
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class QuantumCircuitEnvironment:
    def __init__(self, cpp_simulator):
        self.cpp_simulator = cpp_simulator
        
    def reset(self):
        # C++シミュレータから初期状態を取得
        gates_list = self.cpp_simulator.get_initial_gates()
        circuit_state = self.cpp_simulator.get_circuit_state()
        return gates_list, circuit_state
        
    def step(self, action):
        # アクションをC++シミュレータに送信して実行
        next_gates_list = self.cpp_simulator.execute_action(action)
        next_circuit_state = self.cpp_simulator.get_circuit_state()
        reward = -self.cpp_simulator.get_execution_time()  # 実行時間を負の報酬として使用
        done = self.cpp_simulator.is_circuit_complete()
        
        return (next_gates_list, next_circuit_state), reward, done

def train_model(episodes, batch_size=32):
    cpp_simulator = setup_cpp_simulator()  # C++シミュレータとの接続を設定
    env = QuantumCircuitEnvironment(cpp_simulator)
    
    # 状態空間とアクション空間のサイズを設定
    state_size = get_state_size()  # UniqueTableKeyの次元
    action_size = get_action_size()  # 可能な並列実行の組み合わせ数
    
    agent = QuantumCircuitOptimizer(state_size, action_size)
    
    for episode in range(episodes):
        gates_list, circuit_state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(gates_list, circuit_state)
            (next_gates_list, next_circuit_state), reward, done = env.step(action)
            
            agent.remember(
                gates_list, circuit_state, action, reward,
                next_gates_list, next_circuit_state, done
            )
            
            gates_list, circuit_state = next_gates_list, next_circuit_state
            total_reward += reward
            
            if done:
                print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}")
                break
                
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
        if episode % 10 == 0:
            agent.update_target_model()