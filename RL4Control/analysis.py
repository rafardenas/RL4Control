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
from datetime import datetime
time = datetime.now().strftime('%m%d_%H%M')
sys.path.append('../')

path = 'RL4Control/Assets/Q_learning_0106_1421_agent.pkl'
trained_agent = pickle.load(open(path,'rb')) #load the agent (change the name)

params   = {'u_m' : 0.0923*0.62, 'K_N' : 393.10, 'u_d' : 0.01, 'Y_nx' : 504.49}       
steps_   = np.array([10])       #or movements                                                    
tf       = 16.*24                                                 
#x0       = np.array([0.5,150.0]) #have to make sure initial state is also multiple of the grid                                                                # initial conditions of environment
x0       = np.array([0.5,150.])
modulus  = np.array([0.05, 10]) #granularity of the state∫s discretisation                                                                 # line distance of state space
state_UB = np.array([5, 1000])  # state space upper bound

# Agent definitions: num_actions, eps_prob, alpha, discount
num_actions = 15                                # number (range) of actions available to agent

disc1 = np.array([0.85])                        # discount factor in back-allocation
disc2 = np.array([0.95])                        # discount factor in agent learning
xi_ = np.array([0.5])                             # Epsilon greedy definition (from experiment)

# Experiment defintions: env, agent, controls, episodes
controls          = np.linspace(0,7,num_actions)       # defining possible control actions
#assert episodes_train > 1000, "Number of training epochs must be a factor of 1000 for plotting purposes"
episodes_valid    = 500                                  # number of validation epochs
reward_validation = np.zeros((episodes_valid, xi_.shape[0], disc1.shape[0], disc2.shape[0]))    # memory allocation 
bracket           = int(1000)

# running experimentmin(rewards[0])
def validation_iter():
    "One iteration of validation experiment"
    env                 = Model1(params, steps_, tf, x0, controls)            
    validation          = validation_experiment(env, trained_agent, controls, episodes_valid, 0, noise=True)
    reward_validation   = validation.simulation()
    return validation, reward_validation


exper_number = 1                                  #number of validation runs

rewards = dict.fromkeys(range(exper_number), [])
valid_objects = []

for i in range(exper_number):
    validation, reward_validation = validation_iter()
    rewards[i] = reward_validation
    valid_objects.append(validation)

#plotting training validation results
agent_name = str(trained_agent.learning_algo) + "_" + str(time)
#plot_rew_validation(rewards, agent_name, show = True, save=False)
plot_violin_actions(validation, agent_name , save=False)







