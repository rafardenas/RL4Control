#Running experiments
#28.12.2020

import numpy as np
import matplotlib.pyplot as plt
from agents import Monte_Carlo
from envs import Model1
from Experiment import Experiment

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
xi_ = np.array([0.2])                             # Epsilon greedy definition (from experiment)

# Experiment defintions: env, agent, controls, episodes
controls          = np.linspace(0,7,num_actions)       # defining possible control actions
episodes_train    = 20000                         # number of training epochs
#assert episodes_train > 1000, "Number of training epochs must be a factor of 1000 for plotting purposes"
episodes_valid    = 1                                  # number of validation epochs
reward_training   = np.zeros((episodes_train, xi_.shape[0], disc1.shape[0], disc2.shape[0]))      # memory allocation 
reward_validation = np.zeros((episodes_valid, xi_.shape[0], disc1.shape[0], disc2.shape[0]))    # memory allocation 
bracket           = int(1000)

## These function are for plotting the output 
def EpochNoMean(data, bracket):
    nrows           = int(data.shape[0]/bracket)
    plot_prep_mean  = np.zeros((int(nrows)))
    for f in range(0,nrows):
        x = data[f*bracket:f*bracket+ bracket-1]
        y = np.mean(x,0)
        plot_prep_mean[f] = y
    return plot_prep_mean

#plot of 1000 epoch mean throughout training
def Plotting(data, nrows, bracket, pNo_mean, agent_name):
    plt.figure(figsize =(15,7.5))
    plt.scatter(np.linspace(0,len(data),nrows), data, label= 'Mean R over 1000 epochs')
    plt.xlabel('Training epochs (1e3)',  fontsize=28)
    plt.ylabel('Mean reward over ' + str(bracket)+ ' epochs', fontsize=28)
    plt.tick_params(labelsize=24)
    plt.show()
    #plt.savefig('insert your computer path here' + str(pNo_mean) + '_' + str(agent_name) +'.png')

# running experiment
env                 = Model1(params, steps_, tf, x0, controls)                  # calling environment
agent               = Monte_Carlo(steps_, num_actions, modulus, state_UB, lr = 1, disc1 = 0.85, disc2 = 0.95)                 # calling agent
#agent.printd((1.0, 80.0, 10.0))
experiment          = Experiment(env, agent, controls, episodes_train, xi_)  # calling training experiment
reward_training, d  = experiment.simulation()                                # running training experiment
reward_train_mean   = EpochNoMean(reward_training,bracket)
print(reward_train_mean)
nrows               = int(len(reward_train_mean))
x_o                 = Plotting(reward_train_mean, nrows, bracket, "Time_allocation", agent_name )

