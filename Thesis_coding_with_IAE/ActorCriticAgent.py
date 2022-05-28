
# coding: utf-8

# In[1]:


import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(ActorCriticAgent, self).__init__()
        self.fc1_dims=256
        self.fc2_dims=256
        self.fc1 = nn.Linear(*input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.pi = nn.Linear(self.fc2_dims, n_actions)
        self.v = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda_is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)
        return (pi, v)


# In[3]:


class ActorCriticAgent():
    def __init__(self, lr, input_dims, n_actions, gamma=0.99):
        self.gamma = gamma
        self.lr = lr
        self.actor_critic = ActorCriticNetwork(self.lr, input_dims, n_actions)
        self.log_prob = None  # log probability of the most recent action
        
    def get_action(self, state, edge, episode):
        
        #convert numpy arrays to tensors
        stateT = T.tensor([state], dtype=T.float).to(self.actor_critic.device)
        edgeT = T.tensor([edge], dtype=T.float).to(self.actor_critic.device)
        
        #get probabilities from the pi network but ignore the value from the value network using _
        probabilities, _ = self.actor_critic.forward(stateT)
        
        #pass probabilities through softmax
        probabilities = F.softmax(probabilities, dim=1)
        action_probs = T.distributions.Categorical(probabilities)
        
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.log_prob = log_prob
        return action.item() # dereference action because its a tensor
    
    def learn(self, state, reward, newState, terminal):
        
        self.actor_critic.optimizer.zero_grad()
        stateT = T.tensor([state], dtype=T.float).to(self.actor_critic.device)
        newStateT = T.tensor([newState], dtype=T.float).to(self.actor_critic.device)
        rewardT = T.tensor([reward], dtype=T.float).to(self.actor_critic.device)
        
        _, critic_state_value = self.actor_critic.forward(stateT)
        _, critic_new_state_value = self.actor_critic.forward(newStateT)
        
        delta = reward + self.gamma*critic_new_state_value*(1-int(done)) - critic_state_value
        
        actor_loss = -self.log_prob*delta
        critic_loss = delta**2
        
        (actor_loss + critic_loss).backward()
        self.actor_critic_.optimizer.step()

