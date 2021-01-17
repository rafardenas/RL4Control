# Plotting functions
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import scipy.integrate as scp
import numpy as np
import matplotlib.pyplot as plt
import time




def plot_rew_validation(rewards, name, show = True, save = False):
    plt.figure(figsize=(5,5))
    for i in range(len(rewards.keys())):
        plt.plot(np.arange(len(rewards[i])), rewards[i], label =  "Validation #: " + str(i), linewidth = 1)
        plt.title("Validation reward")
        plt.xlabel("Episode Number")
        plt.ylabel("Reward")
        plt.ylim(bottom = min(rewards[i]))
        plt.legend()
    _pre = list(map(lambda x: x, rewards.values()))
    minima = min(rewards[0])
    if minima < -500:
        minima = -500 
    plt.yticks(np.arange(minima, max(rewards[0]), 20))
    avg_rew = np.array([np.array(i) for i in _pre]).mean()
    plt.axhline(y= avg_rew, c = "red", label = "Average Rew")
    plt.legend()   
    if show:
        plt.show()
    if save:    
        plt.savefig('RL4Control/Assets/' + str(name) + '_val_reward.png')

def plot_epsilon(experiment, show = True):
    epss = experiment.epsilons
    plt.figure(figsize=(5,5))
    plt.plot(np.arange(len(epss)), epss)
    plt.title("Epsilon")
    plt.xlabel("Episode Number")
    if show:
        plt.show()

def plot_max_rew(experiment, show = True, save = False, xi = 1):
    max_rewards = experiment.max_rewards
    plt.figure(figsize=(5,5))
    plt.plot(np.arange(len(max_rewards)), max_rewards)
    plt.title("Max reward")
    plt.xlabel('episodes')
    plt.ylabel('max reward')
    plt.title("Training. eps_decay = {}".format(xi), fontsize=20)
    if show:
        plt.show()
    if save:    
        plt.savefig('RL4Control/Assets/' + 'max_rew.png')

def plot_min_rew(experiment, show = True):
    min_rewards = experiment.min_rewards
    plt.figure(figsize=(5,5))
    plt.plot(np.arange(len(min_rewards)), min_rewards)
    plt.title("Min reward")
    if show:
        plt.show()

def plot_violin_actions(validation, name, show = True, save = False):
    control_actions = validation.control_actions
    fig, ax1 = plt.subplots(figsize=(5,5))
    ax1.set_title('Control actions')
    ax1.set_ylabel('Nitrate inflow rate')
    ax1.violinplot(control_actions)
    if show:
        plt.show()
    if save:    
        fig.savefig('RL4Control/Assets/' + str(name) + '_violin_actions.png')


##################################################
##############deep RL funcions
##################################################


def plot_loss_function(loss, show = True):
    fig, ax1 = plt.subplots(figsize=(5,5))
    ax1.set_title('Loss')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss')
    ax1.plot(loss)
    if show:
        plt.show()

def plot_reward(eps_rewards, show = True):
    fig, ax1 = plt.subplots(figsize=(5,5))
    ax1.set_title('Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.plot(eps_rewards, label = 'Episodic Reward')
    avg_rew = np.array(eps_rewards).mean()
    ax1.axhline(y= avg_rew, c = "red", label = "Average Rew")
    plt.legend() 
    if show:
        plt.show()
    

