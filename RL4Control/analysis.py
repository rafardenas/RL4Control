#Running validation results


import os
import gym
import sys
import numpy as np
import matplotlib.pyplot as plt
from envs import Model1
from validation import validation_experiment
import pickle
from utils import *
sys.path.append('../')


trained_agent = pickle.load(open('RL4Control/Assets/agent.pkl','rb')) #load the agent

params   = {'u_m' : 0.0923*0.62, 'K_N' : 393.10, 'u_d' : 0.01, 'Y_nx' : 504.49}       
steps_   = np.array([10])       #or movements                                                    
tf       = 16.*24                                                 
#x0       = np.array([0.5,150.0]) #have to make sure initial state is also multiple of the grid                                                                # initial conditions of environment
x0       = np.array([0.5,150.])
modulus  = np.array([0.05, 10]) #granularity of the state∫s discretisation                                                                 # line distance of state space
state_UB = np.array([5, 1000])  # state space upper bound

# Agent definitions: num_actions, eps_prob, alpha, discount
agent_name = 'MonteCarlo'
num_actions = 15                                # number (range) of actions available to agent

disc1 = np.array([0.85])                        # discount factor in back-allocation
disc2 = np.array([0.95])                        # discount factor in agent learning
xi_ = np.array([0.5])                             # Epsilon greedy definition (from experiment)

# Experiment defintions: env, agent, controls, episodes
controls          = np.linspace(0,7,num_actions)       # defining possible control actions
#assert episodes_train > 1000, "Number of training epochs must be a factor of 1000 for plotting purposes"
episodes_valid    = 100                                  # number of validation epochs
reward_validation = np.zeros((episodes_valid, xi_.shape[0], disc1.shape[0], disc2.shape[0]))    # memory allocation 
bracket           = int(1000)

# running experiment
env                 = Model1(params, steps_, tf, x0, controls)            
validation          = validation_experiment(env, trained_agent, controls, episodes_valid, 0)
reward_validation   = validation.simulation()

#plotting training validation results

_ = plot_rew_validation(reward_validation)
#print(validation.control_actions)
plot_violin_actions(validation)





