
# coding: utf-8

# In[1]:


from collections import deque
import collections
import numpy as np
import random

# for building DQN model
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam


# In[2]:


class DTDQNAgent():
    def __init__(self, env):        
        self.state_size = -1 #env.getNumberOfPoints()**2
        self.action_size = -1 #12
        self.env = env
        # hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.0001        
        self.epsilon_max = 1.0
        self.epsilon_decay = 0.00006
        self.epsilon_min = 0.00000001
        
        self.batch_size = 32    
        
        # create replay memory using deque
        self.memory = deque(maxlen=2000)             
    
    def reset(self):
        self.memory.clear()  
        
    def initializeModel(self, env):
        self.state_size = env.getNumberOfPoints()**2
        self.action_size = 13
        self.model = self.build_model()
        
    def initializeModelForValidation(self, env, modelWeightsFileName):
        self.state_size = env.getNumberOfPoints()**2
        self.action_size = 13
        self.model = self.build_model()
        self.model.load_weights(modelWeightsFileName)
        
    # approximate Q function using Neural Network
    def build_model(self):
        model = Sequential()
        model.add(Dense(2084, input_dim = self.state_size + self.action_size, activation ='relu'))
        model.add(Dense(1024,activation ='relu'))
        model.add(Dense(512,activation ='relu'))
        model.add(Dense(256,activation ='relu'))
        model.add(Dense(128,activation ='relu'))
        model.add(Dense(1, activation ='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        model.summary
        return model
    
    # save sample <s,a,r,s'> to the replay memory 
    def append_sample(self, state, action, reward, next_state, terminal_state):
        self.memory.append((state, action, reward, next_state, terminal_state))
    
    def prediction(self, state, poss_actions):
        X_test = np.zeros((len(poss_actions), self.state_size + self.action_size))
        for i in range(len(poss_actions)):
            #print('poss_action: {}', poss_actions[i])
            dummy = self.env.getStateActionEncoding(state, poss_actions[i])
            X_test[i,:] = dummy
        prediction = self.model.predict(X_test)
        prediction = prediction.reshape(len(poss_actions))
        return prediction
    
    def get_action(self, state, edge, episode):
    # get action from model using epsilon-greedy policy
    # Decay in ε after we generate each sample from the environment
        poss_actions = self.env.getPossibleActions(edge)
        epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay*episode)
            
        if not poss_actions:
            return [], epsilon
        
        if np.random.rand() <= epsilon: # Exploration: randomly choosing and action      
            ptChosenForAction = np.random.choice(range(len(poss_actions)))
            action = poss_actions[ptChosenForAction]
        else: #Exploitation: this gets the action corresponding to max q-value of current state
            q_values = self.prediction(state, poss_actions)
            action_index = np.argmax(q_values)
            action = poss_actions[action_index]            
        return action, epsilon
    
    def get_action_for_validation(self, state, edge):
        poss_actions = self.env.getPossibleActions(edge)
        if not poss_actions:
            return []
        
        q_values = self.prediction(state, poss_actions)
        action_index = np.argmax(q_values)
        action = poss_actions[action_index]            
        return action
    
    # pick samples randomly from replay memory (with batch_size) and train the network
    def train_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from the memory
        mini_batch = random.sample(self.memory, self.batch_size)
        update_input = np.zeros((self.batch_size, self.state_size + self.action_size))
        #update_output = np.zeros((self.batch_size, self.action_size))
        update_output = {}
        actions, rewards, terminal_states = [], [], []        
        
        for i in range(self.batch_size):
            
            state, action, reward, next_state, terminal_state = mini_batch[i]
            
            newStateActions = self.env.getPossibleActionsForNewState(state, next_state)            
            actions.append(action)
            rewards.append(reward)
            terminal_states.append(terminal_state)            

            update_input[i] = self.env.getStateActionEncoding(state, action)          
            #print('num actions new state: {}', newStateActions)
            #print('next state: {}'.format(next_state))
            
            if len(newStateActions) == 0:
                update_output[i] = [0]
            else:
                update_output[i]= self.prediction(next_state, newStateActions)
        
        target = np.zeros((self.batch_size))
            
        # get your target Q-value on the basis of terminal state
        for i in range(len(terminal_states)):
            if terminal_states[i]:
                target[i] = rewards[i]
            else:
                target[i] = rewards[i] + self.discount_factor * (np.amax(update_output[i]))

        self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)
    
    def load(self, name):
        self.model.load_weights(name)
                    
    def save(self, name):
        self.model.save_weights(name)

