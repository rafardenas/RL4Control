#Deep Q network implementation
#First part towards the complete implementation of the paper by Mnih (2015)

import logging
import os
import sys
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

#PACKAGE_PARENT = '..'
#SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
#sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
sys.path.append('./')
from config.linearDQN import Lin_config
from envs.test_env import EnvTest
from utils.deep_utils import Linearexp
from utils.plot_utils import *
from utils.utils import *
from utils.xpreplay import xpreplay

#################################################
############Pending: Set 'done' flag for terminal states
##################################################


class NN(nn.Module):
    def __init__(self, batch_size, num_actions, config, decay_func):
        super(NN, self).__init__()
        self.config = config
        self.linear = nn.Linear(batch_size, num_actions, bias=True)
        self.loss_function = nn.functional.mse_loss
        self.optimizer = torch.optim.SGD([{'params' : self.parameters(), 'lr' : 1, 'initial_lr' : 1, 'final_lr' : 0.001}])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, decay_func, last_epoch= 0)

    def forward(self, x):
        output = self.linear(x)
        return output


class DQN():
    def __init__(self, NN1, NN2, env, exploration, batch_size, num_actions, config, replay):
        self.env = env
        self.exploration = exploration
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.config = config
        self.est_net, self.target_net = NN1, NN2
        self.replay = replay


    def encode_state(self, state):
        state = np.reshape(state, (-1)) #change to flatten?
        s = state / self.config.scaler
        
        return s
    
    def get_action(self, net, state, eps):
        p = np.random.uniform()
        state = torch.from_numpy(state).float()
        if p < eps:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                action = torch.argmax(net.forward(state)).item()
        return action

    def act(self):
        "Fill up buffer with some initial transitions, random acting"
        for i in range(self.config.learning_start):
            s = self.env.reset()
            s = self.encode_state(s)
            done = False
            while not done:
                action = self.env.action_space.sample()
                n_s, r, done, _ = self.env.step(action)
                n_s = self.encode_state(n_s)
                self.replay.store_sequence(s, action, r, n_s)
                s = n_s
        return


    def training_step(self, t, decay = False):
        tot_rew = 0
        self.exploration.update(t)
        s = self.env.reset()
        s = self.encode_state(s)
        i = 0
        done = False
        loss = 0
        v = 0
        while done == False:
            i += 1
            action = self.get_action(self.est_net, s, self.exploration.epsilon)
            ns, r, done, _ = self.env.step(action)
            tot_rew += r
            ns = self.encode_state(ns)
            self.replay.store_sequence(s, action, r, ns)
            s = ns
            
        states_batch, actions_batch, rewards_batch, n_s_batch = self.replay.sample_batch()
        
        states_batch = torch.tensor(states_batch).float()
        actions_batch = torch.LongTensor(actions_batch).unsqueeze(-1)
        rewards_batch = torch.tensor(rewards_batch).float()
        n_s_batch = torch.tensor(n_s_batch).float()
        #print(states_batch.shape, actions_batch.shape, rewards_batch.shape, n_s_batch.shape)
        
        """with torch.no_grad():
            max_q = r + self.config.gamma * self.target_net(ns).max(1)[0]
            v = self.est_net(s)[0, action]"""

        with torch.no_grad():
            max_q = rewards_batch + self.config.gamma * self.target_net(n_s_batch).max(1)[0]

        loss = self.est_net.loss_function(max_q.unsqueeze_(-1), self.est_net(states_batch).gather(dim=1, index=actions_batch)).float()
        loss.backward()
        self.est_net.optimizer.step()
        if decay:
            self.est_net.scheduler.step()
        self.est_net.optimizer.zero_grad()

        lr = self.est_net.scheduler._last_lr

        #with torch.no_grad():
        #    self.makedotloss = loss
        
        return loss.item(), max_q.mean().item(), tot_rew, lr      #returning mean loss
    
    def swap(self, t):
        if t % self.config.swap_sch == 0 and t != 0:
            self.target_net.load_state_dict(self.est_net.state_dict())

    def evaluate(self):
        array_total_rew = []
        array_cum_q_value = []
        for i in range(self.config.nsteps_eval):
            total_rew = 0
            actions_t = []
            cum_q_value = 0
            done = False
            s = self.env.reset()
            s = self.encode_state(s)
            #s = torch.from_numpy(s)
            while not done:
                action = self.get_action(self.est_net, s, self.config.soft_epsilon)
                ns, r, done, _ = self.env.step(action)
                v = self.est_net(torch.from_numpy(s).float())
                #print(v)
                #print(actions_t)
                actions_t.append(action)
                ns = self.encode_state(ns)
                #ns = torch.from_numpy(ns)
                total_rew += r
                cum_q_value += self.est_net(torch.from_numpy(ns).float()).max(-1)[0]
                s = ns

            array_total_rew.append(total_rew)
            array_cum_q_value.append(cum_q_value.detach().numpy())

        return array_total_rew, array_cum_q_value



class DQN_experiment(DQN):
    def __init__(self, NN1, NN2, env, exploration, batch_size, num_actions, config, replay):
        super().__init__(NN1, NN2, env, exploration, batch_size, num_actions, config, replay)
        self.est_net, self.target_net = NN1, NN2
        self.env = env
        self.exploration = exploration
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.config = config

    def run_training(self, decay = False):
        
        #explor_sch = self.exploration(self.config.eps_begin, self.config.eps_end, self.config.eps_nsteps)
        #deepQ = self.Network(self.est_net, self.target_net, self.env, explor_sch, self.batch_size, self.num_actions, self.config)
        losses = []
        q_values = []
        training_rew = []
        lrs = []
        self.act()
        for t in range(self.config.nsteps_train):
            self.swap(t)
            loss, max_q, tot_reward, lr = self.training_step(t, decay)
            losses.append(loss)
            q_values.append(max_q)
            training_rew.append(tot_reward)
            lrs.append(lr)
        print(self.est_net.scheduler._last_lr)

        return losses, q_values, training_rew, lrs

    def run_validation(self):
        rewards_eval, q_vals_ev = self.evaluate()   #validation
        return rewards_eval, q_vals_ev

        
env = EnvTest((5, 5, 1))
lr_func = lambda epoch: ((Lin_config.lr_end - Lin_config.lr) / Lin_config.lr_nsteps) * epoch + Lin_config.lr if epoch < Lin_config.lr_nsteps else 0.001
#lr_func = lambda epoch: 0.95 * epoch
#lr_func = lambda epoch: 0.001 + np.exp(100/epoch)
Net1, Net2 = NN(25,5, Lin_config, lr_func), NN(25,5, Lin_config, lr_func)
explor_sch = Linearexp(Lin_config.eps_begin, Lin_config.eps_end, Lin_config.eps_nsteps)
#lr_schedule = Linearexp(Lin_config.lr, Lin_config.lr_end, )
#lr_schedule.update()

batch_size = 1
num_actions = 5
replay = xpreplay(Lin_config.buffer_size, Lin_config.batch_size)


DQNexp = DQN_experiment(Net1, Net2, env, explor_sch, batch_size, num_actions, Lin_config, replay)

#tr_losses, tr_q_values, training_rew, lrs = DQNexp.run_training(decay=True)
#print(tr_losses)
#print(tr_q_values)
#print(training_rew)
#rewards_eval, q_vals_ev = DQNexp.run_validation()


#plot_loss_function(tr_losses)
#plot_reward(training_rew)
#plot_reward(rewards_eval)
#plot_loss_function(lrs)










    



    
