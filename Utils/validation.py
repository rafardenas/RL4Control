#4.01.2020
#Validation experiments while training
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from utils import *
import scipy.integrate as scp
import numpy as np


class validation_experiment(object):
    """Core script of the simulation. Depoloying the agent in the environment validate the learned policy
    Params:
        agent(object): Trained agent
        env(object): Model of the environment that generates the response
        controls(array): available control values for the manipulated variables
        episodes(int): Number of training epochs
        xi(int): Rate of decay for epsilon value. See utils.py
        """
    def __init__(self, env, agent, controls, episodes, xi, noise=False):
        self.env = env
        self.agent = agent
        self.controls = controls
        self.episodes = episodes
        self.xi = xi
        self.noise = noise
        movements = int(self.env.tf/float(self.env.dt))
        self.control_actions = np.zeros((episodes * movements))

    def simulation(self):
        r"""
        Using defined environment. In a loop until the end of the episode, using the learned policy
        imparts control action from e-greedy policy and gets a reward from the environment.
        """
        dt, movements, x0   = self.env.dt, int(self.env.tf/float(self.env.dt)), self.env.x0
        model, ctrls        = self.env.response, self.controls
        episodes            = self.episodes
        control_actions     = self.control_actions
        print(control_actions.shape)

        xt      = np.zeros((movements+1, x0.shape[0] + 1, episodes))  
        tt      = np.zeros((movements+1))
        c_hist  = np.zeros((movements, episodes))
        ctrl    = np.zeros((movements, episodes))
        reward  = np.zeros((episodes))
        
        for ei in range(episodes):        
            current_state = tuple((*tuple(x0), 1)) 
            xt[0,:,ei]    = np.array(list(current_state))                                 
            tt[0]         = 0.

            #eps_decay(self.xi, ei, episodes)
            eps_decay = 0                               #for validation always act greedy
            
            for s in range(movements):                                           
                action_indx  = self.agent.act_greedy(self.agent.d, tuple(current_state)) 
                ctrl[s,ei]   =  ctrls[action_indx]                               
                c_hist[s,ei] = action_indx                                       
                ode          = scp.ode(model)                           
                ode.set_integrator('lsoda', nsteps=3000)                         
                ode.set_initial_value(current_state[:x0.shape[0]],dt)
                ode.set_f_params(ctrl[s,ei])                                     
                current_state = list(ode.integrate(ode.t + dt))

                noisy = np.random.normal(0, 0.125, size = len(current_state)) if self.noise else 0                  
                current_state = discrete_env(np.array(current_state) + noisy, self.agent.modulus, s + 2, self.agent.state_UB)
                xt[s+1,:,ei]  = current_state                                    
                tt[s+1]       = (s+1)*dt          
                control_actions[ei * s] = ctrls[action_indx]
                
            for i in [0, 0.2, 0.4, 0.6, 0.8]:
                if i == ei/episodes:
                    print('Validation is', i*100 , ' percent complete')
            r = self.env.reward(xt[:,:,ei]) 
            reward[ei] = np.sum(r)
        return reward          



