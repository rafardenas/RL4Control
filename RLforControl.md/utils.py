#RL for control, utils
#27.12.2020
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

import scipy.integrate as scp
import numpy as np

def discrete_env(state, modulus, s, stochasticity = True):
    """
    Discretisation of the system
    """  
    resid = state % modulus     
    resid = resid/modulus       
    UB = 1 - resid              
    
    if stochasticity:
        draw = np.random.randint(0,2, state.shape[0])
        for i in range(state.shape[0]):
            if draw[i] < UB[i]:
                state[i] = state[i] - resid[i] * modulus[i]
            else:
                state[i] = state[i] - resid[i] * modulus[i] + modulus[i]
    else:
        state[i] = state[i] - resid[i] * modulus[i]

    #Rouding precision errors for purpose of key values
    for i in range(len(state)):
        if state[i] < 0:
            state[i] = 0
        elif state[i] < modulus[i]/2:
            state[i] = 0
    

        f = str(modulus[i])
        decimal = f[::-1].find('.')  
        state[i] = np.round(state[i], decimal)

    state = (*tuple(state), s)

    return state


def integrator(model, current_state, dt, ctrl, modulus, s):
    """
    Util function for integration of the dynamics
    """
    ode = scp.ode(model)                           # define ode, using model defined as np array
    ode.set_integrator('lsoda', nsteps=3000)                         # define integrator
    ode.set_initial_value(current_state,dt)                          # set initial value
    ode.set_f_params(ctrl[s])             #aka params                        # set control action
    
    current_state = ode.integrate(ode.t + dt)
    #current_state = discrete_env(np.array(current_state), modulus, s + 2)
    #current_state = current_state[:2]
    #print("finished 1")
    #print("{},{}".format(ctrl[s], current_state))
    #print(current_state)
    return current_state




def eps_decay(xi, ei, episodes):
    """
    Params:
        xi: decay constant. Set 1 for no decay
        ei: episode index, used for decay progress
        episodes: number of total episodes. (the decay rate will approach 0.1 when completion)
    """
    if xi == 1:
        return 1

    G = -np.log(0.1) * xi * episodes
    if ei < G:
        behave = np.exp(-ei / (episodes * xi))
    else:
        behave = 0.1
    
    return behave

    
from envs import Model1
params   = {'u_m' : 0.0923*0.62, 'K_N' : 393.10, 'u_d' : 0.01, 'Y_nx' : 504.49}
steps_   = np.array([10]) 
tf       = 16.*24
x0       = np.array([0.5,150.])
modulus  = np.array([0.05, 10])
num_actions = 15
controls          = np.linspace(0,7,num_actions) 


env = Model1(params, steps_, tf, x0, controls) 
dt = tf / steps_

ctrl = [1.5 ,6. , 0.5, 1.5 ,5.5 ,5.,  5.,  5. , 5.5 ,6. ]

states= [x0]
current_state = x0
for i in range(steps_[0]):
    state = integrator(env.response, current_state, dt, ctrl, modulus, i)
    current_state = state
    states.append(current_state)
    

