import os
import gym
import numpy as np
import logging
import time
import sys

import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from deep_utils import Linearexp
from test_env import EnvTest
from config.linearDQN import Lin_config
from Utils.utils import *
from Utils.plot_utils import *



class NN(nn.Module):
    def __init__(self, batch_size, num_actions, config):
        super(NN, self).__init__()
        self.config = config
        self.linear = nn.Linear(batch_size, num_actions, bias=True)
        self.loss_function = nn.functional.mse_loss
        self.optimizer = torch.optim.SGD(self.parameters(), lr = self.config.lr)
    
    def forward(self, x):
        output = self.linear(x)
        return output

    
class DQN():
    def __init__(self, NN1, NN2, env, exploration, batch_size, num_actions, config):
        self.env = env
        self.exploration = exploration
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.config = config
        self.est_net, self.target_net = NN1, NN2


    def encode_state(self, state):
        state = np.reshape(state, (1, -1))
        s = state / self.config.scaler
        s = torch.from_numpy(s).float()
        return s
    
    def get_action(self, net, state, eps):
        p = np.random.uniform()
        if p < eps:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                action = torch.argmax(net.forward(state)).item()
        return action


    def training_step(self, t):
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
            with torch.no_grad():
                ns = self.encode_state(ns)
                max_q = r + self.config.gamma * self.target_net(ns).max(1)[0]
                v = self.est_net(s)[0, action]


            loss += self.est_net.loss_function(max_q[0], self.est_net(s)[0, action])
            s = ns
        loss.backward()
        
        self.est_net.optimizer.step()
        self.est_net.optimizer.zero_grad()

        with torch.no_grad():
            self.makedotloss = loss
        
        return (loss / i), max_q, tot_rew      #returning mean loss
    
    def swap(self, t):
        if t % self.config.swap_sch == 0 and t != 0:
            self.target_net.load_state_dict(self.est_net.state_dict())

    def evaluate(self):
        array_total_rew = []
        array_cum_q_value = []
        for i in range(self.config.nsteps_eval):
            total_rew = 0
            cum_q_value = 0
            done = False
            s = self.env.reset()
            s = self.encode_state(s)
            while not done:
                action = self.get_action(self.est_net, s, self.config.soft_epsilon)
                ns, r, done, _ = self.env.step(action)
                v = self.est_net(s)
                ns = self.encode_state(ns)
                total_rew += r
                cum_q_value += self.est_net(ns).max(1)[0]
                s = ns

            array_total_rew.append(total_rew)
            array_cum_q_value.append(cum_q_value.detach().numpy())

        return array_total_rew, array_cum_q_value



class DQN_experiment(DQN):
    def __init__(self, NN1, NN2, env, exploration, batch_size, num_actions, config):
        super().__init__(NN1, NN2, env, exploration, batch_size, num_actions, config)
        self.est_net, self.target_net = NN1, NN2
        self.env = env
        self.exploration = exploration
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.config = config

    def run_training(self):
        
        #explor_sch = self.exploration(self.config.eps_begin, self.config.eps_end, self.config.eps_nsteps)
        #deepQ = self.Network(self.est_net, self.target_net, self.env, explor_sch, self.batch_size, self.num_actions, self.config)
        losses = []
        q_values = []
        training_rew = []
        for t in range(self.config.nsteps_train):
            self.swap(t)
            loss, max_q, tot_reward = self.training_step(t)
            losses.append(loss)
            q_values.append(max_q)
            training_rew.append(tot_reward)

        
        return losses, q_values, training_rew

    def run_validation(self):
        rewards_eval, q_vals_ev = self.evaluate()   #validation
        return rewards_eval, q_vals_ev

        
env = EnvTest((5, 5, 1))
Net1, Net2 = NN(25,5, Lin_config), NN(25,5, Lin_config)
explor_sch = Linearexp(Lin_config.eps_begin, Lin_config.eps_end, Lin_config.eps_nsteps)
batch_size = 1
num_actions = 5

DQNexp = DQN_experiment(Net1, Net2, env, explor_sch, batch_size, num_actions, Lin_config)

tr_losses, tr_q_values, training_rew = DQNexp.run_training()
rewards_eval, q_vals_ev = DQNexp.run_validation()


plot_loss_function(tr_losses)
plot_reward(training_rew)

plot_reward(rewards_eval)










    



    
