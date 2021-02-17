#Experiment Function Approx
#Core script for simulation, define inputs such as number of control actions, starting state, granularity of the discretisation (modulus), etc.
import os
import pickle
import sys

import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as scp
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from torch import nn
sys.path.append('./')
#importing auxiliary functions and classes
#PACKAGE_PARENT = '..'
#SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
#sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from envs.env_DRL import Model2
from utils.deep_utils import Linearexp
from utils.xpreplay import xpreplay


class Experiment:
    """
    Core script of the simulation. Depoloying the agent in the environment to act and learn from it
    Params:
        agent(object): Agent to train
        env(object): Model of the environment that generates the response
        controls(array): available control values for the manipulated variables
        episodes(int): Number of training epochs
        xi(int): Rate of decay for epsilon value. See utils.py
        noise (bool): Whether or not to introduce noise in the states transitions
    """
    def __init__(self, env, agent, config, exploration, replay, noise=False):
        self.env             = env
        self.agent           = agent
        self.controls        = config.controls
        self.num_actions     = config.num_actions
        self.episodes        = config.tr_episodes
        self.exploration     = exploration
        self.noise           = noise
        self.config          = config
        self.control_actions = np.zeros(shape = (self.episodes, self.num_actions + 1))
        self.movements       = int(self.env.tf/float(self.env.dt))
        self.actions         = []
        self.replay          = replay 

    def get_action(self, state):
        p = np.random.uniform()
        if p < self.exploration.epsilon:
            action = np.random.choice(len(self.controls))
        else:
            action = torch.argmax(self.agent.target_net(state))
        return action

    def swap(self, t):
        if t % self.config.swap_sch == 0 and t != 0:
            self.agent.target_net.load_state_dict(self.agent.est_net.state_dict())

    def act(self):
        "Fill up buffer with some initial transitions, random acting"
        for i in range(self.config.learning_start):
            state = self.env.reset()
            for s in range(self.movements + 1):
                action = self.get_action(state)       
                ns, r, time_step = self.env.transition(state, action)
                self.replay.store_sequence(state, action, r, ns)
                state = ns
        return

    def training_step(self, ei):
        r"""
        Training step DQN
        """     
        state = self.env.reset() 
        loss = 0
        tot_rew = 0
        for s in range(self.movements + 1):
            action = self.get_action(state)       #eps_prob is epsilon
            ns, r, time_step = self.env.transition(state, action)         #have to use timestep for completeness flag
            self.actions.append(self.controls[action].round(1))
            tot_rew += r
            #print(r)
            with torch.no_grad():                                 #maximize a scalar?
                max_q = r + self.config.gamma * self.agent.target_net(ns).max().flatten(-1)
                v = (self.agent.est_net(state)[action]).flatten(-1)
            loss += self.agent.est_net.loss_function(max_q, (self.agent.est_net(state)[action]).flatten(-1))
            state = ns
        
        loss.backward()
        self.agent.est_net.optimizer.step()
        self.agent.est_net.optimizer.zero_grad()

        with torch.no_grad():
            self.makedotloss = loss

        return loss.item()/self.num_actions, self.control_actions, tot_rew / self.num_actions    
    
    def training_step_batch(self, ei):
        r"""
        Training step bath with replay DQN
        """     
        state = self.env.reset() 
        loss = 0
        tot_rew = 0
        
        for s in range(self.movements + 1):
            action = self.get_action(state)       #eps_prob is epsilon
            ns, r, time_step = self.env.transition(state, action)         #have to use timestep for completeness flag
            self.actions.append(self.controls[action].round(1))
            #a = self.controls[action].round(1)
            tot_rew += r
            self.replay.store_sequence(state, action, r, ns)
            state = ns
            
        s_batch, a_batch, r_batch, ns_batch = self.replay.sample_batch()

        s_batch = torch.tensor(s_batch).float()
        #print(s_batch)
        a_batch = torch.LongTensor(a_batch).unsqueeze(-1)
        #print(a_batch)
        r_batch = torch.tensor(r_batch).float()
        #print(r_batch)
        ns_batch = torch.tensor(ns_batch).float()
        #print(ns_batch)
            
        with torch.no_grad():                                 #maximize a scalar?
            max_q = r_batch + self.config.gamma * self.agent.target_net(ns_batch).max(1)[0]
            #print(max_q)
            v = self.agent.est_net(s_batch).gather(dim=1, index=a_batch)
            #print(v)
            
        loss = self.agent.est_net.loss_function(max_q.unsqueeze_(-1), self.agent.est_net(s_batch).gather(dim=1, index=a_batch))#.float()
        #print(loss)
        loss.backward()
        
        self.agent.est_net.optimizer.step()
        self.agent.est_net.optimizer.zero_grad()

        with torch.no_grad():
            self.makedotloss = loss

        return loss.item(), self.control_actions, tot_rew / self.num_actions   



    def train(self):
        losses = []
        actions = []
        rews = []
        self.act()
        for ei in range(self.config.tr_episodes):
            self.swap(ei)
            self.exploration.update(ei)
            #loss, control_dist, avg_rew = self.training_step(ei)
            loss, control_dist, avg_rew = self.training_step_batch(ei)
            losses.append(loss)
            actions.append(control_dist)
            rews.append(avg_rew)
        return losses, actions, rews, self.actions 
    
class NN_use_case(nn.Module):
    def __init__(self, batch_size, num_actions):
        super(NN_use_case, self).__init__()
        #self.linear = nn.Linear(batch_size, num_actions, bias=True)
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, num_actions)
        
        self.loss_function = nn.functional.mse_loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)          #[{'params' : self.parameters(), 'lr' : 0.005}])    #, 'initial_lr' : 1, 'final_lr' : 0.001}])
        #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, decay_func, last_epoch= 0)

    def forward(self, x):
        x = torch.tensor(x).float()
        a1 = F.relu(self.fc1(x))
        out = self.fc2(a1)
        return out

class DQN_agent:
    def __init__(self, NN1, NN2):
        self.est_net    = NN1
        self.target_net = NN2

class use_case_config():
    #general params
    parameters = {'u_m' : 0.0923*0.62, 'K_N' : 393.10, 'u_d' : 0.01, 'Y_nx' : 504.49}
    num_actions      = 10
    batch_size       = 32
    steps = steps_   = np.array([10])
    tf               = 16.*24               
    x0               = np.array([0.5,150.]) 
    controls         = np.linspace(0,7,num_actions)
    modulus          = np.array([0.05, 10]) 
    state_UB         = np.array([5, 1000])

    #general
    tr_episodes         = 1
    swap_sch            = 1000
    gamma               = 0.95
    buffer_size         = 1000
 
    #exploration and learning
    start_epsilon    = 1
    end_epsilon      = 0.1
    n_steps          = tr_episodes / 2 
    lr               = 0.005
    lr_end           = 0.001
    lr_nsteps        = tr_episodes / 2
    learning_start   = 200 

c           = use_case_config()
batch_size  = c.batch_size
num_actions = c.num_actions

explore     = Linearexp(c.start_epsilon, c.end_epsilon, c.n_steps)
env         = Model2(c.parameters, c.steps, c.tf, c.x0, c.controls, c.modulus, c.state_UB)\
#lr_func    = lambda epoch: ((c.lr_end - c.lr) / c.lr_nsteps) * epoch + c.lr if epoch < c.lr_nsteps else 0.001
lr_func     = 1 
Net1, Net2  = NN_use_case(batch_size, num_actions), NN_use_case(batch_size, num_actions)
agent       = DQN_agent(Net1, Net2)
replay      = xpreplay(c.buffer_size, c.batch_size)
exper       = Experiment(env, agent, c, explore, replay)


#uncomment to run
"""
loss, control_actions, avg_reward, a = exper.train()
#print(loss, control_actions, avg_reward)
#plt.plot(avg_reward)
#print(loss)
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15, 7.5))
ax1.plot(loss)
ax2.plot(avg_reward)
plt.show()
actions = np.array(a)

def plot_violin_actions(data):
    control_actions = data
    fig, ax1 = plt.subplots(figsize=(5,5))
    ax1.set_title('Control actions')
    ax1.set_ylabel('Nitrate inflow rate')
    ax1.violinplot(control_actions)
    plt.show()

#print(actions)
plt.hist(actions)
plt.show()

#plot_violin_actions(a)  """
