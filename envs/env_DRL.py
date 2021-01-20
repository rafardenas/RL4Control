#RL for control, environment class V2. 
#The classes contained in envs_tabular.py were refractored to seamless use with Deep RL agents
#19.01.2021 - Rafael C.
import numpy as np
import os
import gym
import sys
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
#importing auxiliary functions
from Utils.utils import discrete_env, eps_decay
import scipy.integrate as scp
import pickle

#Base class for the environment, its children are the dynamics of different reactions, for example.
#attributes: control actions
#methods: 
    #provide the reward
    #apply the action specified by the agent
    #generate transisions: from state to next state 


class Environment2:
    r"""
    Base class for the environment, its children are, e.g. the dynamics of different reactions.
    Attributes:
        params(dic): parameters of the diff equations for the dynamics
        steps(int): number of discrete intervals in one episode (take it as equivalent of 'movements' in Agent's class, aka one movement per step)  
        t_f(int): lenght of the episode (unitless)
        x0(array): initial state, each item is the initial state for a variable. 
        control(dict or tuple): dictionary/tuple holding the values of the controlling variables, those variables are per se the controlling actions
    """
    def __init__(self, parameters, steps, tf, x0, control, modulus, state_UB, noisy = False):
        self.parameters = parameters
        self.steps = steps 
        self.tf = tf
        self.x0 = x0
        self.control = control
        self.noisy = noisy
        self.modulus = modulus
        self.upper_bounds = state_UB
        self.time_step = 0

        
class Model2(Environment2):
    """
    Dynamic model for biomass production, taken from RL discrete tutorial. Invoke it by name for integration
    """
    def __init__(self, parameters, steps, tf, x0, control, modulus, state_UB, noisy = False): #missing state and t?
        super(Model2, self).__init__(parameters, steps, tf, x0, control, modulus, state_UB, noisy = False)
        self.dt = tf/steps
        
    def response(self, t, state, control):  #what is 't'?
        params = self.parameters
        FCn   = control   
        # state vector [Cx, Cn]
        Cx  = state[0]
        Cn  = state[1]
        # parameters for the ODEs, looks like...
        u_m  = params['u_m']
        K_N  = params['K_N']
        u_d  = params['u_d']
        Y_nx = params['Y_nx']
        # variable rate equations aka diff equations below
        dev_Cx  = u_m * Cx * Cn/(Cn+K_N) - u_d*Cx**2
        dev_Cn  = - Y_nx * u_m * Cx * Cn/(Cn+K_N) + FCn
        
        return np.array([dev_Cx, dev_Cn],dtype='float64')

    def reward(self,state):
        "Reward function for Model 1. Nice to have: Be modifiable"
        reward = [100 * s[0] - s[1] for s in state]
        return reward

    def single_reward(self, state):
        "Reward function for Model 1. for a single state"
        reward = 100 * state[0] - state[1]
        return reward

    def reset(self):
        state = self.x0
        self.time_step = 0
        return state

    def transition(self, state, action):
        action = self.control[action]
        #print(state, action)
        self.time_step += 1
        ode   = scp.ode(self.response)                               # define ode, using model defined as np array
        ode.set_integrator('lsoda', nsteps=3000)                     # define integrator
        ode.set_initial_value(state, self.dt)                        # set initial value
        ode.set_f_params(action)                                     # set control action
        next_state = ode.integrate(ode.t + self.dt)                       # integrate system
        noise = np.random.normal(0, 0.125, size = len(next_state)) if self.noisy else 0
        reward = self.single_reward(state)
        next_state = np.array((next_state) + noise).round(1)
        return next_state, reward, self.time_step


    def discrete_env(self, state, time_step, stochasticity = True):
        """
        Discretisation of the state
        """  
        resid = state % self.modulus     
        resid = resid/self.modulus       
        UB = 1 - resid              
        
        if stochasticity:
            draw = np.random.randint(0,2, state.shape[0])
            for i in range(state.shape[0]):
                if draw[i] < UB[i]:
                    state[i] = state[i] - resid[i] * self.modulus[i]
                else:
                    state[i] = state[i] - resid[i] * self.modulus[i] + self.modulus[i]
        else:
            state[i] = state[i] - resid[i] * self.modulus[i]

        #Rouding precision errors for purpose of key values
        for i in range(len(state)):
            if state[i] < 0:
                state[i] = 0
            elif state[i] < self.modulus[i]/2:
                state[i] = 0
            elif state[i] > self.upper_bounds[i]:
                state[i] = self.upper_bounds[i]
        

            f = str(self.modulus[i])
            decimal = f[::-1].find('.')  
            state[i] = np.round(state[i], decimal)

        state = (*tuple(state), time_step)

        return state