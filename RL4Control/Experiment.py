#Experiment
#Core script for simulation, define inputs such as number of control actions, starting state, granularity of the discretisation (modulus), etc.
import os
import gym
import sys
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from utils import discrete_env, eps_decay
import scipy.integrate as scp
import numpy as np
import pickle
sys.path.append('../')

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
    def __init__(self, env, agent, controls, episodes, xi, noise=False):
        self.env = env
        self.agent = agent
        self.controls = controls
        self.episodes = episodes
        self.xi = xi
        self.noise = noise

    def dump(self, agent, name):
        "Save the agent used in the experiment to a pickle file"
        pickle.dump(agent, open('RL4Control/Assets/'+str(name)+'_agent.pkl', 'wb'))
        return

    def simulation(self):
        r"""
        Using defined environment. In a loop until the end of the episode: imparts control action from e-greedy policy and gets a reward from the environment, the agent uses it to learn.
        """
        # internal definitions
        dt, movements, x0   = self.env.dt, int(self.env.tf/float(self.env.dt)), self.env.x0
        model, ctrls        = self.env.response, self.controls       #controls: set of control options i.e. np.linspace(0,7,num_actions)
        episodes            = self.episodes
        self.epsilons       = np.zeros((episodes))
        self.max_rewards    = np.zeros((episodes))
        self.min_rewards    = np.zeros((episodes)) 

        # compile state and control trajectories
        xt      = np.zeros((movements+1, x0.shape[0] + 1, episodes))  #add one to record the time step in state
        tt      = np.zeros((movements+1))
        c_hist  = np.zeros((movements, episodes))
        ctrl    = np.zeros((movements, episodes))
        reward  = np.zeros((episodes))

        
        for ei in range(episodes):        
            current_state = tuple((*tuple(x0), 1)) #start from 0?
            xt[0,:,ei]    = np.array(list(current_state))                                        #current state is a tuple dims(1,n) 
            tt[0]         = 0.
            eps_prob = eps_decay(self.xi, ei, episodes)
            self.epsilons[ei] = eps_prob

        ########################## Running Monte Carlo ################################

            if self.agent.learning_algo == "MC":

                for s in range(movements):                                           #'s' is the movement #, we currently at
                    action_indx  = self.agent.act(self.agent.d, tuple(current_state), eps_prob, s) 
                    ctrl[s,ei]   =  ctrls[action_indx]                               #find control action relevant to index from agent.act
                    c_hist[s,ei] = action_indx                                       # storing control history for each epoch
                    ode          = scp.ode(model)                           # define ode, using model defined as np array
                    ode.set_integrator('lsoda', nsteps=3000)                         # define integrator
                    ode.set_initial_value(current_state[:x0.shape[0]],dt)                          # set initial value
                    ode.set_f_params(ctrl[s,ei])                                     # set control action
                    current_state = list(ode.integrate(ode.t + dt))                  # integrate system
                    noisy = np.random.normal(0, 0.125, size = len(current_state)) if self.noise else 0
                    current_state = discrete_env(np.array(current_state) + noisy, self.agent.modulus, s + 2, self.agent.state_UB)
                    xt[s+1,:,ei]  = current_state                                    # add current state Note: here we can add randomnes as: + RandomNormal noise
                    tt[s+1]       = (s+1)*dt          #tracking in which "time step" we are?
                for i in [0, 0.2, 0.4, 0.6, 0.8]:
                    if i == ei/episodes:
                        print('Simulation is', i*100 , ' percent complete')
            
                r = self.env.reward(xt[:,:,ei]) #[movements, state, episode] for the whole episode
                self.min_rewards[ei] = np.min(r)
                self.max_rewards[ei] = np.max(r)
                reward[ei] = np.sum(r)          #summing for reward of the whole episode

                try:
                    if self.agent.learning_algo == "MC":
                        self.agent.Incremental(xt[:,:,ei], c_hist[:,ei], r)
                    
                    elif self.agent.learning_algo == "FV":
                        self.agent.Learn_FirstVisit(xt[:,:,ei], c_hist[:,ei], r)
                except:
                    print("Algorithm unavailable")
                    break

        ########################## Running Q-learning ################################

            elif self.agent.learning_algo == "Q_learning":          
                
                for s in range(movements - 1):                                           #'s' is the movement #, we currently at
                    action_indx  = self.agent.act(self.agent.d, tuple(current_state), eps_prob, s) 
                    ctrl[s,ei]   =  ctrls[action_indx]                               #find control action relevant to index from agent.act
                    c_hist[s,ei] = action_indx                                       # storing control history for each epoch
                    
                    ode          = scp.ode(model)                           # define ode, using model defined as np array
                    ode.set_integrator('lsoda', nsteps=3000)                         # define integrator
                    ode.set_initial_value(current_state[:x0.shape[0]],dt)                          # set initial value
                    ode.set_f_params(ctrl[s,ei])                                     # set control action
                    
                    next_state = list(ode.integrate(ode.t + dt))                  
                    noisy = np.random.normal(0, 0.125, size = len(next_state)) if self.noise else 0
                    next_state = discrete_env(np.array(next_state) +  noisy, self.agent.modulus, s + 2, self.agent.state_UB)
                    xt[s+1,:,ei]  = next_state                                    # add current state Note: here we can add randomnes as: + RandomNormal noise
                    tt[s+1]       = (s+1)*dt          
                    
                    r = self.env.single_reward(next_state)
                    self.agent.Q_learn(current_state, next_state, action_indx, r)

                    current_state = next_state

                for i in [0, 0.2, 0.4, 0.6, 0.8]:
                    if i == ei/episodes:
                        print('Simulation is', i*100 , ' percent complete')
        

                r = self.env.reward(xt[:,:,ei]) #We dont need to keep stored everything, we do it here for convenience for reporting purposes
                self.min_rewards[ei] = np.min(r)
                self.max_rewards[ei] = np.max(r)
                reward[ei] = np.sum(r)          #summing for reward of the whole episode
            
            else:
                print("Algorithm unavailable")
                break

        d = self.agent.learned(self.agent.d)
        return reward, d 


