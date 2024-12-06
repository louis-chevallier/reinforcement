import os, sys
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
import pickle
import markdown

EKOX(gym.__version__)

#tqdm = lambda x : x

parser = argparse.ArgumentParser(
    prog='DQLearning')


def md(md_txt) :
    fn = md_txt.split()[0]
    rad = fn.split("-")[0]
    html = markdown.markdown(md_txt, extensions=['md4mathjax'])
    with open("out.html", "w", encoding="utf-8", errors="xmlcharrefreplace") as output_file:
        output_file.write(html)
    e = lambda command : os.system(command)
    e("pandoc out.html -o out.pdf")
    e("pdftoppm -f 1 -l 9999 out.pdf -png %s" % rad)
    

md("""
xx-1.png
# running
suivi des courses, comparaison en temps réel de la perf
$x = y²$
## c1
1. aaa
2. bbb

$$x = {-b \pm \sqrt{b^2-4ac} \over 2a}.$$

""")



    
parser.add_argument("--train", action="store_true", default=False)
parser.add_argument("--load", action="store_true", default=False)
parser.add_argument("--start_episodes", type=int, default=0)
parser.add_argument("--episodes", type=int, default=10000)
parser.add_argument("--eval", action="store_true", default=False)
args = parser.parse_args()


# Open the image form working directory
image = np.asarray(Image.open('bw1.png')).copy()

EKOX(image)

EKOI(image.astype(float))
#image[:,:] = 1

EKOX(np.mean(image))
EKOX(image.shape)
EKOX(image)

BLOCK=1

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
    reward = 1 if dist < 2./W else 0
    if dist < 2./W :
        reward = 1 #+ 1. / count 
    else :
        reward = 0

    
    truncated = dist > 2 or count > W*2
    terminated = dist < 2/W
    out_of_bounds = s[0] > 1 or s[0] < 0 or s[1] > 1 or s[1] < 0
    terminated |= out_of_bounds
    if not out_of_bounds :
        collide = image[int(s[0]*H), int(s[1]*W)] == BLOCK
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
        self.pp = np.ones((H,W,3))
        self.pp[:,:,0] = image

    def clear(self) :
        self.pp = np.ones((H,W,3))
        self.pp[:,:,0] = image
        
    def reset(self) :
        
        while (True) :
            DD = W//2
            ee = np.random.randint(DD, size=(2)) - DD//2
            #EKOX(ee)
            self.state = np.asarray((H/2, W/2)) + ee
            s = self.state
            #EKON(s, image[int(s[0]), int(s[1])])
            collide = image[int(s[0]), int(s[1])] == BLOCK
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
        y,x = min(int(s[0]*H), H-1), min(int(s[1]*W), W-1)
        self.pp[y, x, 1:] = (v, v)

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

def epsilon_func(episode, total_episode) :
    a, b = -1/total_episode, 1
    ee = a * episode + b
    res = max(0.1, ee)
    return res

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
    rr = torch.rand(1)
    #EKON(rr, epsilon, rr < epsilon)
    if rr < epsilon:
        # random : p = epsilon
        res = torch.randint(env.action_space.n, (1, 1))
    else:
        # greedy : p = 1-epsilon
        av = q_network(state).detach()
        res = torch.argmax(av, dim=-1, keepdim=True)
    return res

def dqn_training(
        cb,
        q_network: DQNetworkModel,
        policy,
        episodes,
        alpha=0.0001,
        batch_size=64,
        gamma=0.99,
        start_episodes = 0
):
    optim = torch.optim.AdamW(q_network.parameters(), lr=alpha)
    memory = ReplayMemory()
    stats = {'MSE Loss': [], 'Returns': []}
    EKO()
    e10 = episodes//10
    for episode in tqdm(range(start_episodes, episodes + 1 + start_episodes)):
        #EKON(episode , epsilon_func(episode, episodes))
        if episode % e10 == 0 :
            EKO()
            cb(episode)

        #EKO()
        state = env.reset().to(device)
        truncated, terminated = False, False # initiate the terminated and truncated flags
        ep_return = 0
        #EKOX(state)        
        while not truncated and not terminated:
            action = policy(state, epsilon_func(episode, episodes)) # select action based on epsilon greedy policy
            next_state, reward, truncated, terminated, _ = env.step(action) # take step in environment
            #EKON(state.detach().cpu().numpy(), terminated.item(), truncated.item(), action.item())
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

        #epsilon = max(0, epsilon - 1/10000)
        #EKOX(epsilon)
        if episode % 10 == 0:
            target_q_network.load_state_dict(q_network.state_dict())
    EKO()
    return stats    

if CP :

    d = {
        "Acrobot-v1" : "rgb_array",
        "CartPole-v1" : "human" }
    
    env = gym.make(env_name,  render_mode = None  if args.train else d[env_name])
    env = PreProcessEnv(env)



if args.load :
    q_network.load_state_dict(torch.load('qlearning.cpt', weights_only=True))
    
if args.train :
    EKO()
    def cb(num) :
        states = env1.states_samples()
        #EKOX(states)
        rr = np.zeros((H,W))    
        for state in states :
            state = torch.tensor(np.asarray(state)[None, ...]).float()
            q = q_network(state.to(device))
            qq = q.detach().cpu().numpy()
            #EKON(state, q)
            action = torch.argmax(q)
            action_i = int(action.cpu().detach())
            v = float(qq[0,action_i])
            ss = state[0].cpu().detach().numpy()
            vv = lambda xx : min(xx, W-1)
            #EKON(vv(int(ss[0]*H)), vv(int(ss[1]*W)), v)
            rr[vv(int(ss[0]*H)), vv(int(ss[1]*W))] = v
        #EKOX(rr)
        rr = np.where(image == 1, 0, rr)
        EKOI(rr)
        plt.imsave("f_%06d.png" % num, rr);# plt.show()
    EKO()
    d = dqn_training(cb, q_network, policy, args.episodes, start_episodes=args.start_episodes)
    EKOX(len(d["MSE Loss"]))
    with open("loss.pickle","wb") as fd :
        pickle.dump(d["MSE Loss"], fd)
    #plt.imsave("loss.png", d["MSE Loss"]); 
    #plt.plot(d["MSE Loss"]); plt.show()
    #plt.imsave("f_%04d.png" % num, rr);# plt.show()
    
    torch.save(q_network.state_dict(), 'qlearning.cpt')
    state = env.reset()

if args.eval : 
    q_network.eval()
    for i in range(20):
        state = env.reset()
        env1.clear()
        #EKOX(state)
        n=0
        terminated, truncated = False, False
        while not terminated and not truncated:
            with torch.inference_mode():
                action = torch.argmax(q_network(state.to(device)), dim=-1, keepdim=True)
                action_i = int(action.cpu().detach())
                EKON(action, action_i)
                v = float(q_network(state.to(device))[0,action_i].cpu())
                v = 0.5
                state, reward, terminated, truncated, info = env.step(action)
                EKON(state.detach().cpu().numpy(), terminated.item(), truncated.item(), action_i)
                if terminated or truncated :
                    EKON(n, terminated, truncated)
                else :
                    try :
                        env1.paint(v)
                    except Exception as e :
                        EKOX(e)
                        pass

                n += 1
        try :
            env1.display()
        except :
            pass
