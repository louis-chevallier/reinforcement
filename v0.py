import itertools
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
from numba import jit
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import graphviz 
EKOX(gym.__version__)

#tqdm = lambda x : x

parser = argparse.ArgumentParser(
    prog='DQLearning')

    
parser.add_argument("--train", action="store_true", default=False)
args = parser.parse_args()


# Open the image form working directory
image = np.asarray(Image.open('bw1.png')).copy()

EKOX(image)

image[:,:] = 1

EKOX(np.mean(image))
EKOX(image.shape)

CP = False
H,W = image.shape

ground = np.zeros((H,W))


aa = lambda x : np.asarray(x)

d_action = np.asarray([
    aa((0,1)),
    aa((0,-1)),
    aa((1,0)),
    aa((-1,0))
    ])
EKOX(d_action)


#@jit
def xstep(action, s, count) :
    v = d_action[int(action)]        
    #EKON(s, v)
    next_state = s + v / W
    #self.state = next_state
    #EKOX(self.state)        
    dist = np.sqrt((s[0] ** 2 + s[1] **2))
    #reward = float(1. / max(1/100, dist))
    reward = 1 if dist < 2/W else 0
    truncated = dist > 2 or count > W*2
    terminated = dist < 2/W
    out_of_bounds = s[0] > 1 or s[0] < 0 or s[1] > 1 or s[1] < 0
    terminated |= out_of_bounds
    if not out_of_bounds :
        collide = image[int(s[0]*H), int(s[1]*W)] == 0
        terminated |= collide
    #EKON(terminated, truncated, reward, collide)
    #EKOX(image[s[0], s[1]])
    return next_state, reward, int(truncated), int(terminated), None



T = lambda next_state, reward=0 : (next_state, reward)
TERMINATED = -1
G = [
    { 0 : T(TERMINATED), 1 : T(1,1)},
    { 0 : T(2), 1 : T(TERMINATED, 1) },
    { 0 : T(3), 1 : T(4,1) },
    { 0 : T(TERMINATED), 1 : T(TERMINATED, 1) },
    { 0 : T(TERMINATED, 1), 1 : T(TERMINATED, 0) }
]


if False :
    NG = nx.DiGraph()

    #for i, nd in enumerate(G) :    NG.add_node(i)

    el = {}
    nl = {}
    for i, nd in enumerate(G) :
        if nd[0][0] >= 0 :
            NG.add_edge(i, nd[0][0])
            el[(i, nd[0][0])] = nd[0][1]
        if nd[1][0] >= 0 :        
            NG.add_edge(i, nd[1][0])
            el[(i, nd[1][0])] = nd[1][1]        

    nx.draw_networkx_labels(
        NG,
        pos=nx.spring_layout(NG))

    nx.draw_networkx_edge_labels(
        NG,
        pos=nx.spring_layout(NG),
        edge_labels=el,
        font_color='red'
    )
    nx.draw(NG)
    #nx.draw(NG, pos=nx.spring_layout(NG))  # use spring layout
    limits = plt.axis("off")  # turn off axis
    plt.show()


class GraphEnv(gym.Env) :

    def states_samples(self) :
        return [ [e] for e in range(0,4)]

    def __init__(self) :
        self.observation_space = np.ones(shape=(1,1))
        EKOX(self.observation_space)
        self.action_space = Namespace(**{ "n" : 2})
        self.s = 0


        dot = graphviz.Digraph("gg")

        for i, nd in enumerate(G) :
            dot.node(str(i), str(i))
            def e(x) :
                if nd[x][0] >= 0 :        
                    dot.edge(str(i), str(nd[x][0]), label="A_" + str(x))
                else :
                    dot.edge(str(i), "T", label="A_" + str(x))                    
            e(0)
            e(1)
        EKOX(dot)
        dot.render("graph.png", view=True)
    def reset(self) :
        self.state = np.zeros(1)
        self.s = 0
        r = self.state[None, ]
        return r

    def display(self) :
        pass

    def paint(self, v) :
        pass
        
    def step(self, action) :
        self.s += 1
        _is = int(self.state[0])
        sd = G[_is]
        next_state = np.asarray([sd[action][0]])
        #EKON(_is, sd, action, next_state)
        self.state = next_state
        reward = sd[action][1]
        truncated = False
        terminated = next_state == TERMINATED
        return next_state, reward, truncated, terminated, None


class CustomEnv(gym.Env) :

    def states_samples(self) :
        bounds = [ (0, 1), (0, 1) ]
        lb = [ np.linspace(mn, mx, W) for i, (mn, mx) in enumerate(bounds)]
        return list(itertools.product(*lb))        

    def __init__(self) :
        self.observation_space = np.ones(shape=(2,1))
        EKOX(self.observation_space)
        self.action_space = Namespace(**{ "n" : 4})
        self.count = 0
        self.pp = np.ones((H,W,3))
        self.pp[:,:,0] = image
        
    def reset(self) :
        while (True) :
            DD = W/5
            ee = np.random.randint(DD, size=(2)) - DD//2
            self.state = np.asarray((H/2, W/2)) + ee
            s = self.state
            collide = image[int(s[0]), int(s[1])] == 0
            if not collide :
                break
        #EKOX(self.state)            
        self.state /= (H,W)

        self.count = 0
        r = self.state[None, ]
        #EKOX(self.state)
        return r

    def display(self) :
        plt.imshow(self.pp); plt.show()

    def paint(self, v) :
        s = self.state
        self.pp[int(s[0]*H), int(s[1]*W), 1:] = (v, v)

    def step(self, action) :
        self.count += 1
        aaa = xstep(action, self.state, self.count)
        next_state, reward, truncated, terminated, _ = aaa
        self.state = next_state
        return aaa

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

        #pos, velo, angle, ang_velo = obs[0].detach().numpy()
        #EKON(pos, velo, angle, ang_velo, float(reward), int(terminated), int(truncated))
        #if reward < 1. : sys.exit(0)
        
        return obs, reward, terminated, truncated, info

    def reset(self):
        obs = self.env.reset()
        obs = torch.from_numpy(obs[0]).unsqueeze(0).float()
        return obs
    
env_name = "CartPole-v1"
#env_name = "Acrobot-v1"
    
if CP :
    env1 = gym.make(env_name)
else :
    env1 = CustomEnv()
    #env1 = GraphEnv()
    
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
EKO()
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
    batch_size=64,
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
        "CartPole-v1" : "human" }
    
    env = gym.make(env_name,  render_mode = None  if args.train else d[env_name])
    env = PreProcessEnv(env)

if args.train :
    d = dqn_training(q_network, policy, 1000)
    EKOX(len(d["MSE Loss"]))
    #plt.plot(d["MSE Loss"]); plt.show()
    torch.save(q_network.state_dict(), 'qlearning.cpt')

    state = env.reset()
    
    states = env1.states_samples()
    EKOX(states)
    for state in states :
        state = torch.tensor(np.asarray(state)[None, ...]).float()
        q = q_network(state.to(device))
        qq = q.detach().cpu().numpy()
        EKON(state, q)
        action = torch.argmax(q)
        action_i = int(action.cpu().detach())
        v = float(qq[0,action_i].cpu())
        ss = state[0].cpu().detach().numpy()
        EKOX(int(ss[0]*H))
        EKOX(int(ss[1]*W))

    
else : 
    q_network.load_state_dict(torch.load('qlearning.cpt', weights_only=True))
    q_network.eval()
    for i in range(20):
        state = env.reset()
        #EKOX(state)
        n=0
        terminated, truncated = False, False
        while not terminated and not truncated:
            with torch.inference_mode():
                action = torch.argmax(q_network(state.to(device)))
                action_i = int(action.cpu().detach())
                #EKOX(action_i)
                v = float(q_network(state.to(device))[0,action_i].cpu())
                v = 0.5
                state, reward, terminated, truncated, info = env.step(action)
                #EKOX(state)
                try :
                    env1.paint(v)
                except Exception as e :
                    EKOX(e)
                    pass
                if terminated or truncated :
                    EKON(n, terminated, truncated)
                n += 1
        try :
            env1.display()
        except :
            pass
