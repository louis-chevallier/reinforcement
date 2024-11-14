import copy
from utillc import *
import torch
from torch import nn
from torch.functional import F
import argparse
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace

import gymnasium as gym

EKOX(gym.__version__)

#tqdm = lambda x : x

parser = argparse.ArgumentParser(
    prog='DQLearning')

    
parser.add_argument("--train", action="store_true", default=False)
args = parser.parse_args()



CP = False
    


D = 100
ground = np.zeros((D, D))
class CustomEnv(gym.Env) :
    def __init__(self) :
        self.observation_space = np.ones(shape=(2,1))
        EKOX(self.observation_space)
        self.action_space = Namespace(**{ "n" : 4})
        self.s = 0
    def reset(self) :

        self.state = np.asarray((D//2, D//2))
        self.s = 0
        return [self.state]
    def step(self, action) :
        self.s += 1
        v = {
            0 : (1, 0),
            1 : (0, 1),
            2 : (-1, 0),
            3 : (0, -1)
        }[int(action)]
        s = self.state
        #EKON(s, v)
        next_state = np.asarray([s[0] + v[0], s[1] + v[1]])
        self.state = next_state
        reward = float(D - np.sqrt((s[0] ** 2 + s[1] **2)))
        truncated = self.s > D*2
        terminated = next_state[0] == 0 and next_state[1] == 0
        return next_state, reward, truncated, terminated, None

class PreProcessEnv(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        
    def step(self, action):
        action = action.item()
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = torch.from_numpy(obs).unsqueeze(0).float()
        reward = torch.tensor(reward).view(-1, 1)
        terminated = torch.tensor(terminated).view(-1, 1)
        truncated = torch.tensor(truncated).view(-1, 1)

        return obs, reward, terminated, truncated, info

    def reset(self):
        obs = self.env.reset()
        obs = torch.from_numpy(obs[0]).unsqueeze(0).float()
        return obs
    

if CP :
    env = gym.make("CartPole-v1")
else :
    env = CustomEnv()    
env = PreProcessEnv(env)

class DQNetworkModel(nn.Module):
    def __init__(self, in_channels, out_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_classes),
        )

    def forward(self, x):
        return self.layers(x)


device = "cpu"
EKON(env.observation_space.shape[0], env.action_space.n)
q_network = DQNetworkModel(env.observation_space.shape[0], env.action_space.n).to(device)

target_q_network = copy.deepcopy(q_network).to(device).eval()

class ReplayMemory:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def insert(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition

        
        self.position = (self.position + 1) % self.capacity

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        transitions = random.sample(self.memory, batch_size)
        batch = zip(*transitions)
        return [torch.cat([item for item in items]) for items in batch]

def policy(state, epsilon):
    if torch.rand(1) < epsilon:
        return torch.randint(env.action_space.n, (1, 1))
    else:
        av = q_network(state).detach()
        return torch.argmax(av, dim=-1, keepdim=True)

def dqn_training(
    q_network: DQNetworkModel,
    policy,
    episodes,
    alpha=0.0001,
    batch_size=32,
    gamma=0.99,
    epsilon=1,
):
    optim = torch.optim.AdamW(q_network.parameters(), lr=alpha)
    memory = ReplayMemory()
    stats = {'MSE Loss': [], 'Returns': []}
    
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        truncated, terminated = False, False # initiate the terminated and truncated flags
        ep_return = 0
        while not truncated and not terminated:
            action = policy(state, epsilon) # select action based on epsilon greedy policy
            next_state, reward, truncated, terminated, _ = env.step(action) # take step in environment
            
            memory.insert([state, action, reward, truncated,terminated, next_state]) #insert experience into memory
            
            if memory.can_sample(batch_size):
                state_b, action_b, reward_b, truncated_b,terminated_b, next_state_b = memory.sample(batch_size) # sample a batch of experiences from the memory
                qsa_b = q_network(state_b).gather(1, action_b) # get q-values for the batch of experiences
                
                next_qsa_b = target_q_network(next_state_b) # get q-values for the batch of next_states using the target network
                next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0] # select the maximum q-value (greedy)
                
                target_b = reward_b + ~(truncated_b + terminated_b) * gamma * next_qsa_b # calculate target q-value 
                
                loss = F.mse_loss(qsa_b, target_b) # calculate loss between target q-value and predicted q-value

                q_network.zero_grad()
                loss.backward()
                optim.step()
                
                stats['MSE Loss'].append(float(loss))
                
            state = next_state
            ep_return += reward.item()
            
        
        stats['Returns'].append(ep_return)

        epsilon = max(0, epsilon - 1/10000)
        
        if episode % 10 == 0:
            target_q_network.load_state_dict(q_network.state_dict())

    return stats    

if CP :
    env = gym.make("CartPole-v1", render_mode = "human")
    env = PreProcessEnv(env)

if args.train :
    d = dqn_training(q_network, policy, 100)
    EKOX(len(d["MSE Loss"]))
    plt.plot(d["MSE Loss"]); plt.show()
    torch.save(q_network.state_dict(), 'qlearing.cpt')
else : 
    q_network.load_state_dict(torch.load('qlearing.cpt', weights_only=True))
    q_network.eval()
    for i in range(20):
        state = env.reset()
        terminated, truncated = False, False
        while not terminated and not truncated:
            with torch.inference_mode():
                action = torch.argmax(q_network(state.to(device)))
                state, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated :
                    EKON(terminated, truncated)
