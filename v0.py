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
from PIL import Image
import gymnasium as gym

EKOX(gym.__version__)

#tqdm = lambda x : x

parser = argparse.ArgumentParser(
    prog='DQLearning')

    
parser.add_argument("--train", action="store_true", default=False)
args = parser.parse_args()


# Open the image form working directory
image = np.asarray(Image.open('bw.png'))

EKOX(np.mean(image))
EKOX(image.shape)

CP = True #False
D, _ = image.shape

ground = np.zeros((D, D))
class CustomEnv(gym.Env) :
    def __init__(self) :
        self.observation_space = np.ones(shape=(2,1))
        EKOX(self.observation_space)
        self.action_space = Namespace(**{ "n" : 4})
        self.s = 0
    def reset(self) :
        self.pp = np.ones((D,D,3))
        self.pp[:,:,0] = image
        self.state = np.asarray((D/2/D, D/2/D))
        self.s = 0
        r = self.state[None, ]
        #EKOX(self.state)
        return r

    def display(self) :
        plt.imshow(self.pp); plt.show()

    def paint(self, v) :
        s = self.state
        #EKON(v, s)
        self.pp[int(s[0]), int(s[1]), 1:] = (v, v)
        
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
        next_state = np.asarray([s[0] + v[0]/D, s[1] + v[1]/D])
        self.state = next_state
        #EKOX(self.state)        
        dist = np.sqrt((s[0] ** 2 + s[1] **2))
        reward = float(1. / max(1/100, dist))
        truncated = dist > 2
        terminated = dist < 1/100
        collide = image[int(s[0]*D), int(s[1]*D)] == 0
        terminated |= collide
        terminated |= s[0] > 1
        terminated |= s[0] < 0
        terminated |= s[1] > 1
        terminated |= s[1] < 0
        #EKON(terminated, truncated, reward, collide)
        #EKOX(image[s[0], s[1]])
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
    
env_name = "CartPole-v1"
env_name = "Acrobot-v1"
    
if CP :
    env1 = gym.make(env_name)
else :
    env1 = CustomEnv()    
env = PreProcessEnv(env1)

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
#            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
EKON(env.observation_space.shape[0], env.action_space.n)
q_network = DQNetworkModel(env.observation_space.shape[0], env.action_space.n).to(device)

target_q_network = copy.deepcopy(q_network).to(device).eval().to(device)

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
        return [torch.cat([item.to(device) for item in items]) for items in batch]

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
        state = env.reset().to(device)
        truncated, terminated = False, False # initiate the terminated and truncated flags
        ep_return = 0
        #EKOX(state)        
        while not truncated and not terminated:
            action = policy(state, epsilon) # select action based on epsilon greedy policy
            next_state, reward, truncated, terminated, _ = env.step(action) # take step in environment
            next_state = next_state.to(device)
            memory.insert([state, action, reward, truncated,terminated, next_state]) #insert experience into memory
            
            if memory.can_sample(batch_size):
                state_b, action_b, reward_b, truncated_b,terminated_b, next_state_b = memory.sample(batch_size) # sample a batch of experiences from the memory

                #EKOX(state_b)
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

    d = {
        "Acrobot-v1" : "rgb_array",
        "CartPole-v1" ; "human" }
    
    env = gym.make(env_name,  render_mode = None  if args.train else d[env_name])
    env = PreProcessEnv(env)

if args.train :
    d = dqn_training(q_network, policy, 1000)
    EKOX(len(d["MSE Loss"]))
    #plt.plot(d["MSE Loss"]); plt.show()
    torch.save(q_network.state_dict(), 'qlearing.cpt')
else : 
    q_network.load_state_dict(torch.load('qlearing.cpt', weights_only=True))
    q_network.eval()
    for i in range(20):
        state = env.reset()
        n=0
        terminated, truncated = False, False
        while not terminated and not truncated:
            with torch.inference_mode():
                action = torch.argmax(q_network(state.to(device)))
                action_i = int(action.cpu().detach())
                v = float(q_network(state.to(device))[0,action_i].cpu())
                state, reward, terminated, truncated, info = env.step(action)
                try :
                    env1.paint(v)
                except :
                    pass
                if terminated or truncated :
                    EKON(n, terminated, truncated)
                n += 1
        try :
            env1.display()
        except :
            pass
