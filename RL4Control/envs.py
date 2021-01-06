#RL for control, environment class
#27.12.2020
import numpy as np

#Base class for the environment, its children are the dynamics of different reactions, for example.
#attributes: control actions
#methods: generate the reward
    #apply the action specified by the agent
    #One method for each model in the catalog, perhaps can be automized, but later 



class Environment:
    r"""
    Base class for the environment, its children are, e.g. the dynamics of different reactions.
    Attributes:
        params(dic): parameters of the diff equations for the dynamics
        steps(int): number of discrete intervals in one episode (take it as equivalent of 'movements' in Agent's class, aka one movement per step)  
        t_f(int): lenght of the episode (unitless)
        x0(array): initial state, each item is the initial state for a variable. 
        control(dict): dictionary holding the values of the controlling variables, those variables are per se the controlling actions
    """
    def __init__(self, parameters, steps, tf, x0, control):
        self.parameters = parameters
        self.steps = steps 
        self.tf = tf
        self.x0 = x0
        self.control = control
                                #same number of decimals as value dict

        
class Model1(Environment):
    """
    Dynamic model for biomass production, taken from RL discrete tutorial. Invoke it by name for integration
    """
    def __init__(self, parameters, steps, tf, x0, control): #missing state and t?
        super().__init__(parameters = parameters, steps = steps, tf = tf, x0 = x0, control = control)
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

