#Running experiments
#28.12.2020
import os
import gym
import sys
import numpy as np
import matplotlib.pyplot as plt
from agents import Monte_Carlo, Q_Learning
from envs import Model1
from experiment import Experiment
from validation import validation_experiment
from utils import *
from datetime import datetime
time = datetime.now().strftime('%m%d_%H%M')
sys.path.append('../')


params   = {'u_m' : 0.0923*0.62, 'K_N' : 393.10, 'u_d' : 0.01, 'Y_nx' : 504.49}       
steps_   = np.array([10])       #or movements                                                    
tf       = 16.*24                                                 
x0       = np.array([0.5,150.])
modulus  = np.array([0.05, 10]) #granularity of the states discretisation                                                                 # line distance of state space
state_UB = np.array([5, 1000])  # state space upper bound

# Agent definitions: num_actions, eps_prob, alpha, discount
num_actions = 15                                # number (range) of actions available to agent
episodes_train    = 300000

disc1 = np.array([0.85])                        # discount factor in back-allocation
disc2 = np.array([0.95])                        # discount factor in agent learning
xi_ = np.array([0.3])                             # Epsilon greedy definition (from experiment)

# Experiment defintions: env, agent, controls, episodes
controls          = np.linspace(0,7,num_actions)       # defining possible control actions
#assert episodes_train > 1000, "Number of training epochs must be a factor of 1000 for plotting purposes"
reward_training   = np.zeros((episodes_train, xi_.shape[0], disc1.shape[0], disc2.shape[0]))      # memory allocation 
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
def Plotting(data, nrows, bracket, pNo_mean, agent_name, axis = None, xi = None, index = 1, save_plot=False):
    
    #plt.tick_params(labelsize=24)
    if axis != None:
        axis.set_xlabel('Training eps')#,  fontsize=28)
        axis.set_ylabel('avg reward / ' + str(bracket)+ ' epochs')#, fontsize=28)
        axis.set_title("Training. eps_decay = {}".format(xi), fontsize=10)
        axis.scatter(np.linspace(0,len(data),nrows), data, label= 'Mean R over 1000 epochs')
    else:
        plt.figure(figsize =(15,7.5))
        plt.xlabel('Training eps',  fontsize=28)
        plt.ylabel('avg reward / ' + str(bracket)+ ' epochs', fontsize=28)
        plt.title("Training. eps_decay = {}".format(xi), fontsize=20)
        plt.scatter(np.linspace(0,len(data),nrows), data, label= 'Mean R over 1000 epochs')
        plt.tick_params(labelsize=24)
        if save_plot:
            plt.savefig('RL4Control/Assets/' + str(pNo_mean) + '_' + str(agent_name) +'.png')
        plt.show()
    return


# running experiment
def single_run(save_agent = True, save_plot = False):
    env                 = Model1(params, steps_, tf, x0, controls)                  # calling environment
    agent               = Monte_Carlo(steps_, num_actions, modulus, "MC" ,state_UB, lr = 7, disc1 = 0.85, disc2 = 0.95) 
    agent_name = str(agent.learning_algo) + "_" + str(time)
    experiment          = Experiment(env, agent, controls, episodes_train, xi_[0], noise=True)  # calling training experiment
    reward_training, d  = experiment.simulation()                               
    if save_agent:
        experiment.dump(agent, agent_name)                      #saving the agent
    reward_train_mean   = EpochNoMean(reward_training,bracket)
    nrows               = int(len(reward_train_mean))
    Plotting(reward_train_mean, nrows, bracket, agent_name, agent_name, xi=xi_[0], save_plot = save_plot)


single_run(save_agent=False, save_plot= False)


#run the following function to make a 2x2 grid of reward plots based on different decays rates

def grid(save_plot = False):
    decays = [0.1, 0.2, 0.3, 0.4]               #rates of decay for epsilon greedy
    l_rates = [0.5,1,1.5,2]
    idx = -1
    fig, axs = plt.subplots(2, 2, figsize=(10,5))
    for eppss in decays:
        idx += 1
        agent = Q_Learning(steps_, num_actions, modulus, "Q_learning",state_UB, lr = l_rates[idx], disc1 = 0.85, disc2 = 0.95) 
        agent_name = agent.learning_algo
        experiment          = Experiment(env, agent, controls, episodes_train, eppss)  # calling training experiment
        reward_training, d  = experiment.simulation()                                # running training experiment
        reward_train_mean   = EpochNoMean(reward_training,bracket)
        nrows               = int(len(reward_train_mean))
        if idx <= 1:
            ax = axs[0, idx % 2]
        else:
            ax = axs[1, idx % 2]
        Plotting(reward_train_mean, nrows, bracket, "Training", agent_name, axis=ax, xi=[eppss,l_rates[idx]] , index = idx, save_plot = save_plot)
        print("Iteration # {} completed".format(idx + 2))

    plt.subplots_adjust(hspace=0.5)
    plt.savefig('RL4Control/Assets/' + str(1) + '_' + str(agent_name) +'.png')
    plt.show()

#grid()


